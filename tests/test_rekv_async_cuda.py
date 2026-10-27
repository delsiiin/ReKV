"""Snapshot forwards on CPU; real stream/encoder checks run when CUDA is available."""
import copy
from concurrent.futures import ThreadPoolExecutor
import threading
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from transformers import LlamaConfig, LlamaModel, MistralConfig, MistralModel, Qwen2Config, Qwen2Model

from model.rekv.abstract_rekv import Abstract_ReKV
from model.rekv.async_runtime import ReKVAsyncRuntime
from model.rekv.attention.snapshot import LayerSnapshot, SnapshotRetrieval
from model.rekv.patch import patch_hf
from video_qa.rekv_stream_vqa import ReKVStreamVQA


def decoder(config_class=Qwen2Config, model_class=Qwen2Model, device='cpu', fattn=False):
    config = config_class(vocab_size=64, hidden_size=64, intermediate_size=96,
                          num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                          pad_token_id=0, bos_token_id=1, eos_token_id=63)
    return patch_hf(model_class(config).to(device).eval(), n_init=2, n_local=8,
                    topk=2, chunk_size=1, block_size=2, max_cached_block=4,
                    exc_block_size=2, fattn=fattn, async_global_stream=True, pin_memory=True)


class SnapshotForwardTests(unittest.TestCase):
    @torch.inference_mode()
    def test_snapshot_forward_matches_dense_retrieval_and_uses_private_rope(self):
        for config, cls in ((LlamaConfig, LlamaModel), (MistralConfig, MistralModel),
                            (Qwen2Config, Qwen2Model)):
            model = decoder(config, cls)
            layers = []
            for _ in model.layers:
                kv = tuple(torch.randn(1, 1, 6, 32) for _ in range(2))
                layers.append(LayerSnapshot(2, 8, 2, 1, 2, False,
                                            tuple(t[:, :, :0] for t in kv), kv,
                                            (torch.empty(0, 64),), ((),), 0, 6))
            private_rope = copy.deepcopy(model.position_bias)
            contexts = tuple(SnapshotRetrieval(layer) for layer in layers)
            ids = torch.tensor([[7, 8, 9]])
            result = model(ids, past_key_values=contexts, rekv_position_bias=private_rope)
            self.assertIsNone(model.position_bias._cos_cached)
            self.assertIsNotNone(private_rope._cos_cached)
            dense = tuple(layer.retrieve([[0, 1]]) for layer in layers)
            reference = model(ids, past_key_values=dense)
            torch.testing.assert_close(result.last_hidden_state, reference.last_hidden_state)
            for actual, expected in zip(result.past_key_values, dense):
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b)
            self.assertEqual([c.indices for c in contexts], [[[0, 1]], [[0, 1]]])

    def test_all_model_wrappers_forward_snapshot_and_rope_through_generation(self):
        from model.rekv.llava_onevision_rekv import LlavaOneVision_ReKV
        from model.rekv.video_llava_rekv import VideoLlava_ReKV
        from model.rekv.longva_rekv import LongVA_ReKV

        class Tokenizer:
            eos_token_id = 63

            def __call__(self, text):
                return SimpleNamespace(input_ids=[1, 4, 5])

            def decode(self, ids, **kwargs):
                return ','.join(map(str, ids))

        for wrapper in (LlavaOneVision_ReKV, VideoLlava_ReKV, LongVA_ReKV):
            with self.subTest(wrapper=wrapper.__name__), torch.inference_mode():
                model = TinyReKV('cpu')
                model.processor = SimpleNamespace(tokenizer=Tokenizer())
                model.get_input_embeddings = model.language_model.get_input_embeddings
                layers = []
                for _ in model.language_model.layers:
                    kv = tuple(torch.randn(1, 1, 6, 32) for _ in range(2))
                    layers.append(LayerSnapshot(2, 8, 2, 1, 2, False,
                                                tuple(t[:, :, :0] for t in kv), kv,
                                                (torch.empty(0, 64),), ((),), 0, 6))
                rope = copy.deepcopy(model.language_model.position_bias)
                # CPU fixture uses the same contexts, with no CUDA publication event.
                snapshot = SimpleNamespace(layers=layers, position_bias=rope, wait=lambda: None,
                    retrieval_contexts=lambda indices: tuple(SnapshotRetrieval(x, indices) for x in layers))
                answer = wrapper.question_answering(model, {'question': 'q', 'prompt': 'q'},
                                                    max_new_tokens=3, kv_snapshot=snapshot)
                self.assertTrue(answer)
                self.assertIsNone(model.kv_cache)
                self.assertIsNone(model.language_model.position_bias._cos_cached)
                self.assertIsNotNone(rope._cos_cached)


class TinyReKV(torch.nn.Module, Abstract_ReKV):
    def __init__(self, device='cuda', fattn=False):
        torch.nn.Module.__init__(self)
        Abstract_ReKV.__init__(self, None, 2, [1, 2], 8, 2, 1)
        self.language_model = decoder(device=device, fattn=fattn)
        self.lm_head = torch.nn.Linear(64, 64, bias=False).to(device)
        self.eval()

    @property
    def device(self):
        return self.lm_head.weight.device

    def encode_frame(self, index):
        ids = torch.tensor([[3 + index % 20, 4 + index % 20]], device=self.device)
        self.kv_cache = self.language_model(ids, past_key_values=self.kv_cache).past_key_values


@unittest.skipUnless(torch.cuda.is_available(), 'ReKV asynchronous cache checks require CUDA')
class AsyncCUDATests(unittest.TestCase):
    @torch.inference_mode()
    def test_snapshot_qa_leaves_sync_buffers_unallocated_until_sync_retrieval(self):
        model = TinyReKV()
        runtime = ReKVAsyncRuntime(model)
        try:
            with runtime.encoding():
                model.encode_init_prompt()
                for index in range(23):
                    model.encode_frame(index)
                snapshot = runtime.capture()
            with torch.cuda.stream(runtime.answer_stream):
                snapshot.wait()
                expected = tuple(layer.retrieve([[0, 22]]) for layer in snapshot.layers)
            runtime.answer_stream.synchronize()
            with runtime.encoding():
                for cache, reference in zip(model.kv_cache, expected):
                    self.assertIsNone(cache.cuda_cache.data)
                    self.assertEqual(cache.global_buffer.size(-2), cache.n_init)
                    cache.set_retrieved_block_indices([[0, 22]])
                    actual = cache.get_retrieved_kv()
                    for a, b in zip(actual, reference):
                        torch.testing.assert_close(a, b)
                    self.assertIsNotNone(cache.cuda_cache.data)
                    self.assertEqual(cache.global_buffer.size(-2), cache.n_init + cache.topk * cache.block_size)
                    pool, buffer = cache.cuda_cache.data, cache.global_buffer
                    cache.get_retrieved_kv()
                    self.assertIs(cache.cuda_cache.data, pool)
                    self.assertIs(cache.global_buffer, buffer)
                    cache.reset_retrieval()
        finally:
            runtime.close()
            model.clear_cache()

    def test_unbounded_pipeline_encodes_all_frames_before_first_answer_and_matches_sync(self):
        producer_done = threading.Event()

        class PipelineModel(TinyReKV):
            def encode_video(self, frames):
                self.encode_frame(int(frames[0, 0, 0, 0]))

            def get_prompt(self, question):
                return question

            @torch.inference_mode()
            def question_answering(self, inputs, max_new_tokens=3, kv_snapshot=None):
                if kv_snapshot is not None and not producer_done.wait(10):
                    raise TimeoutError('Producer waited for QA despite unbounded mode')
                ids = torch.tensor([[27]], device=self.device)
                past, options = self._retrieve_question(ids, kv_snapshot=kv_snapshot)
                tokens = []
                for _ in range(min(max_new_tokens, 3)):
                    output = self.language_model(ids, past_key_values=past, **options)
                    ids = self.lm_head(output.last_hidden_state[:, -1:]).argmax(-1)
                    tokens.append(ids.item())
                    past = output.past_key_values
                return str(tokens)

        model = PipelineModel()
        sample = dict(video_id='tiny', conversations=[
            dict(question=f'q{i}', answer='', end_time=cutoff)
            for i, cutoff in enumerate((2, 5, 9, 17, 23))])

        def frames(_):
            for i in range(23):
                yield float(i), np.full((1, 1, 3), i, dtype=np.uint8)

        try:
            with tempfile.TemporaryDirectory() as directory:
                records = []
                for asynchronous in (False, True):
                    analyzer = ReKVStreamVQA(anno=[sample], save_dir=directory, sample_fps=1,
                                             qa_model=model, rekv_async=asynchronous,
                                             rekv_async_max_pending_groups=0)
                    analyzer.frame_source = frames
                    if asynchronous:
                        analyzer.finish_encoding = producer_done.set
                    analyzer.analyze_a_video(sample)
                    records.append(analyzer.records)
                    if asynchronous:
                        stats = analyzer.async_video_stats[0]
                        self.assertEqual(stats['peak_pending_groups'], 5)
                        self.assertEqual(stats['backpressure_seconds'], 0.)
                self.assertEqual(records[0], records[1])
        finally:
            producer_done.set()
            model.clear_cache()

    @torch.inference_mode()
    def test_zero_copy_publication_preserves_prefix_through_offload_and_growth(self):
        model = TinyReKV()
        runtime = ReKVAsyncRuntime(model)
        try:
            with runtime.encoding():
                model.encode_init_prompt()
                for i in range(2):
                    model.encode_frame(i)
                before = torch.cuda.memory_allocated(model.device)
                early = runtime.capture()
                self.assertEqual(torch.cuda.memory_allocated(model.device), before)
                self.assertIs(early.layers[0].remainder[0], model.kv_cache[0].global_remainder[0])
                expected = tuple(t.clone() for t in early.layers[0].retrieve([[0, 1]]))
                for i in range(2, 5):
                    model.encode_frame(i)
                published = runtime.capture()
                key_prefix = published.layers[0].block_keys[0]
                self.assertEqual(key_prefix.untyped_storage().data_ptr(),
                                 model.kv_cache[0].block_k[0].data.untyped_storage().data_ptr())
                saved_keys = key_prefix.clone()
                for i in range(5, 23):
                    model.encode_frame(i)
                runtime.answer_stream.wait_stream(runtime.encode_stream)
            with torch.cuda.stream(runtime.answer_stream):
                early.wait()
                published.wait()
                for actual, reference in zip(early.layers[0].retrieve([[0, 1]]), expected):
                    torch.testing.assert_close(actual, reference)
                torch.testing.assert_close(key_prefix, saved_keys)
                self.assertNotEqual(key_prefix.untyped_storage().data_ptr(),
                                    model.kv_cache[0].block_k[0].data.untyped_storage().data_ptr())
                self.assertEqual(len(published.layers[0].blocks[0]), 5)
                with self.assertRaises(IndexError):
                    published.layers[0].blocks[0][5]
        finally:
            runtime.close()
            model.clear_cache()

    @torch.inference_mode()
    def test_live_snapshot_matches_sync_across_offload_and_expansion(self):
        for fattn in (False, True):
            model = TinyReKV(fattn=fattn)
            runtime = ReKVAsyncRuntime(model)
            try:
                with runtime.encoding():
                    model.encode_init_prompt()
                    for frame in range(22):
                        if frame in (0, 2, 5, 17):
                            snapshot = runtime.capture()
                            ids = torch.tensor([[27, 28]], device=model.device)
                            # Save the synchronous reference at exactly this prefix.
                            for cache in model.kv_cache:
                                cache.set_retrieval()
                            reference = model.language_model(ids, past_key_values=model.kv_cache)
                            expected_kv = tuple(tuple(t.clone() for t in pair) for pair in reference.past_key_values)
                            expected_indices = [copy.deepcopy(c.retrieved_block_indices) for c in model.kv_cache]
                            for cache in model.kv_cache:
                                cache.reset_retrieval()
                            model.encode_frame(frame)
                            runtime.answer_stream.wait_stream(runtime.encode_stream)
                            with torch.cuda.stream(runtime.answer_stream):
                                snapshot.wait()
                                contexts = snapshot.retrieval_contexts()
                                result = model.language_model(ids, past_key_values=contexts,
                                                              rekv_position_bias=snapshot.position_bias)
                                for actual, expected in zip(result.past_key_values, expected_kv):
                                    for a, b in zip(actual, expected):
                                        torch.testing.assert_close(a, b)
                                torch.testing.assert_close(result.last_hidden_state, reference.last_hidden_state)
                                self.assertEqual([c.indices for c in contexts], expected_indices)
                                # Generation must remain isolated while live encoding grows.
                                token = torch.tensor([[30]], device=model.device)
                                decoded = model.language_model(token, past_key_values=result.past_key_values,
                                                               rekv_position_bias=snapshot.position_bias)
                                expected = model.language_model(token, past_key_values=expected_kv,
                                                                rekv_position_bias=snapshot.position_bias)
                                torch.testing.assert_close(model.lm_head(decoded.last_hidden_state),
                                                           model.lm_head(expected.last_hidden_state))
                                runtime.answer_stream.synchronize()
                        else:
                            model.encode_frame(frame)
            finally:
                runtime.close()
                model.clear_cache()

    def test_actual_shared_model_two_threads_no_future_frames(self):
        for fattn in (False, True):
            model = TinyReKV(fattn=fattn)
            runtime = ReKVAsyncRuntime(model)
            started, proceed = threading.Event(), threading.Event()
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    with runtime.encoding():
                        model.encode_init_prompt()
                        for i in range(5):
                            model.encode_frame(i)
                        snapshot = runtime.capture()
                        for layer, cache in zip(snapshot.layers, model.kv_cache):
                            self.assertEqual(layer.block_keys[0].untyped_storage().data_ptr(),
                                             cache.block_k[0].data.untyped_storage().data_ptr())

                        def answer(_video, _sample, *, kv_snapshot, **_):
                            kv_snapshot.wait()
                            started.set()
                            if not proceed.wait(10):
                                raise TimeoutError('encoder did not advance')
                            ids = torch.tensor([[27]], device=model.device)
                            contexts = kv_snapshot.retrieval_contexts()
                            output = model.language_model(ids, past_key_values=contexts,
                                                          rekv_position_bias=kv_snapshot.position_bias)
                            # Several forwards overlap with continued main-thread encoding.
                            tokens = []
                            for _ in range(8):
                                token = model.lm_head(output.last_hidden_state[:, -1:]).argmax(-1)
                                tokens.append(token.item())
                                output = model.language_model(token, past_key_values=output.past_key_values,
                                                              rekv_position_bias=kv_snapshot.position_bias)
                            return contexts, tokens

                        future = executor.submit(runtime.answer, answer, {}, {}, snapshot)
                        try:
                            self.assertTrue(started.wait(10))
                            model.encode_frame(5)
                            proceed.set()
                            for i in range(6, 23):
                                model.encode_frame(i)
                        finally:
                            proceed.set()
                    contexts, tokens = future.result(timeout=30)
                    self.assertTrue(all(i < 5 for context in contexts for i in context.indices[0]))
                    self.assertTrue(all(not c.to_retrieve for c in model.kv_cache))
                    self.assertEqual(model.kv_cache[0].num_global_block, 23)
                    # The same frozen prefix produces the same greedy answer after encoding ends.
                    _, reference_tokens = runtime.answer(answer, {}, {}, snapshot)
                    self.assertEqual(tokens, reference_tokens)
            finally:
                proceed.set()
                runtime.close()
                model.clear_cache()


if __name__ == '__main__':
    unittest.main()
