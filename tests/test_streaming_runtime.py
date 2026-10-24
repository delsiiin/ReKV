import os
import unittest
import torch

from model.streaming.cache import StreamingKVCache
from model.streaming.runtime import StreamingConfig, StreamingRuntime


class FakeWorker:
    """Small cache exercising single-process sequential encoding and QA."""
    loads = 0

    def __init__(self, config):
        type(self).loads += 1
        self.config = config
        self.events = []
        self.reset()

    def reset(self):
        self.cache = StreamingKVCache([(torch.zeros(1, 2, 16), torch.zeros(1, 2, 16))]*2,
            self.config['budget_tokens'], 16, group_size=16)

    def append_frame(self, image, timestamp, before_commit=lambda: None):
        self.events.append(('encode', timestamp))
        x = torch.arange(16).view(1, 16, 1).expand(1, 16, 16).half()+float(image)
        self.cache.append_frame(timestamp, [(x, x), (x, x)], score_keys=[x, x])
        if self.cache.next_id % 3 == 0:
            self.close_segment(before_commit)

    def close_segment(self, before_commit=lambda: None):
        before_commit()
        self.cache.close_segment([torch.ones(2, 2, 16)]*2)
        return self.stats()

    def stats(self):
        return dict(self.cache.memory_stats(), pid=os.getpid(),
                    hits={i: f.hits for i, f in self.cache.frames.items()},
                    pinned=len(self.cache.snapshots), model_loads=type(self).loads)

    def answer(self, snapshot, question, on_retrieved):
        self.events.append(('qa_start', snapshot.frames[-1].frame_id))
        if question == 'crash':
            raise RuntimeError('intentional test worker failure')
        k, v, ids, _ = snapshot.retrieve_layer(0, torch.ones(1, 2, 1, 16), topk=64)
        version = snapshot.version
        del snapshot
        on_retrieved(ids)
        self.events.append(('qa_end', ids[-1]))
        return dict(frame_ids=list(ids), tokens=k.shape[-2], version=version, pid=os.getpid(), model_loads=type(self).loads)


class RuntimeTests(unittest.TestCase):
    def config(self):
        return StreamingConfig(budget_tokens=150, device='cpu',
                               group_size=16)

    def setUp(self):
        FakeWorker.loads = 0

    def test_single_load_same_process_and_sequential_events(self):
        from unittest.mock import patch
        with patch('torch.multiprocessing.get_context', side_effect=AssertionError('No child process allowed')), \
                StreamingRuntime(self.config(), FakeWorker) as runtime:
            for i in range(4):
                runtime.append_frame(i, i)
            answer = runtime.ask('what', 3.)
            self.assertEqual(answer['frame_ids'], [0, 1, 2, 3])
            self.assertFalse(runtime.model.cache.snapshots)
            for i in range(4, 8):
                runtime.append_frame(i, i)
            self.assertEqual(runtime.model.events[:7], [
                ('encode', 0), ('encode', 1), ('encode', 2), ('encode', 3),
                ('qa_start', 3), ('qa_end', 3), ('encode', 4)])
            stats = runtime.close_segment()
            self.assertEqual(stats['pid'], os.getpid())
            self.assertEqual(answer['pid'], os.getpid())
            self.assertEqual(runtime.startup['execution_mode'], 'sequential')
            self.assertEqual(stats['model_loads'], 1)
            self.assertEqual(answer['model_loads'], 1)
            self.assertLessEqual(stats['persistent_bytes'], stats['budget_bytes'])
            self.assertGreater(stats['counts']['evicted_frames'], 0)
            answer2 = runtime.ask('again', 7.)
            self.assertIn(7, answer2['frame_ids'])
            self.assertTrue(all(hits <= 2 for hits in runtime.memory_stats()['hits'].values()))
            with self.assertRaises(ValueError):
                runtime.ask('past', 1.)
            model = runtime.model
            runtime.reset()
            self.assertIs(runtime.model, model)
            self.assertEqual(runtime.memory_stats()['retained_frames'], 0)
            self.assertEqual(runtime.memory_stats()['model_loads'], 1)
        self.assertIsNone(runtime.model)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            runtime.ask('closed')

    def test_split_devices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'one device'):
            StreamingConfig(encode_device='cuda:0', qa_device='cuda:1')
        self.assertEqual(StreamingConfig(encode_device='cpu', qa_device='cpu').device, 'cpu')

    def test_inference_failure_releases_snapshot_and_closes_runtime(self):
        runtime = StreamingRuntime(self.config(), FakeWorker)
        try:
            runtime.append_frame(0, 0.)
            cache = runtime.model.cache
            with self.assertRaisesRegex(RuntimeError, 'intentional test worker failure'):
                runtime.ask('crash')
            self.assertTrue(runtime.closed)
            self.assertIsNone(runtime.model)
            self.assertFalse(cache.snapshots)
        finally:
            runtime.close()


class ModelBridgeTests(unittest.TestCase):
    @torch.inference_mode()
    def test_qwen_ragged_layers_native_summary_and_encoding(self):
        from transformers import Qwen2Config, Qwen2Model
        from model.patch import patch_hf
        from model.streaming.attention import streaming_attention_forward, EncodeLayer, RetrieveLayer
        from model.streaming.onevision import native_decoder
        model = Qwen2Model(Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)).half().eval()
        patch_hf(model, attention_factory=streaming_attention_forward, n_local=64, n_init=2, fattn=False)
        initial = model(input_ids=torch.tensor([[1, 2]]), use_cache=True)
        sink = [(k[0], v[0]) for k, v in initial.past_key_values]
        store = StreamingKVCache(sink, 10000, 4, group_size=8)
        contexts = [EncodeLayer(past) for past in initial.past_key_values]
        model(input_ids=torch.tensor([[3, 4, 5, 6]]), past_key_values=contexts, use_cache=True)
        self.assertTrue(all(x.captured is not None for x in contexts))
        for i in range(3):
            store.append_frame(i, [x.captured for x in contexts],
                               score_keys=[x.score_key for x in contexts], rope_start=contexts[0].position_start)
        store.close_segment([torch.ones(2, 4, 8), -torch.ones(2, 4, 8)])
        from dataclasses import replace
        frame = store.frames[0]
        # Deliberately give two layers different cache lengths.
        store.frames[0] = replace(frame, layers=(frame.layers[0].drop_low().shrink(8), frame.layers[1]))
        snapshot = store.acquire_snapshot(1)
        hits = set()
        retrieved = model(input_ids=torch.tensor([[7, 8]]),
            past_key_values=[RetrieveLayer(snapshot, i, 3, hits) for i in range(2)], use_cache=True)
        self.assertEqual(hits, {0, 1, 2})
        self.assertNotEqual(retrieved.past_key_values[0][0].shape[-2], retrieved.past_key_values[1][0].shape[-2])
        decoded = model(input_ids=torch.tensor([[9]]), past_key_values=retrieved.past_key_values, use_cache=True)
        self.assertTrue(torch.isfinite(decoded.last_hidden_state).all())
        with native_decoder(model):
            native = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
            self.assertTrue(torch.isfinite(native.last_hidden_state).all())
        again = model(input_ids=torch.tensor([[9]]), past_key_values=retrieved.past_key_values, use_cache=True)
        torch.testing.assert_close(again.last_hidden_state, decoded.last_hidden_state)


class SequentialModelTests(unittest.TestCase):
    @torch.inference_mode()
    def test_encoding_and_native_summary_between_questions_preserve_model_behavior(self):
        from types import SimpleNamespace
        from transformers import Qwen2Config, Qwen2Model
        from model.patch import patch_hf
        from model.streaming.onevision import OneVisionEngine, native_decoder
        from model.streaming.attention import streaming_attention_forward
        torch.manual_seed(19)
        decoder = Qwen2Model(Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)).half().eval()
        patch_hf(decoder, attention_factory=streaming_attention_forward, n_local=8, n_init=2, fattn=False)
        worker = object.__new__(OneVisionEngine)
        worker.decoder, worker.device, worker.frame_tokens = decoder, torch.device('cpu'), 4
        worker.config = dict(topk=3, max_new_tokens=5, recent_frames=2)
        worker.model = SimpleNamespace(lm_head=torch.nn.Linear(32, 32, bias=False).half().eval())
        class Tokenizer:
            eos_token_id = 100
            def __call__(self, *args, **kwargs):
                return SimpleNamespace(input_ids=torch.tensor([[7, 8]]))
            def decode(self, tokens, **kwargs):
                return str(tokens)
        worker.tokenizer = Tokenizer()
        initial = decoder(input_ids=torch.tensor([[1, 2]]), use_cache=True)
        sink = [(k[0], v[0]) for k, v in initial.past_key_values]
        cache = StreamingKVCache(sink, 10000, 4, group_size=8)
        for i in range(4):
            key = torch.randn(2, 4, 8).half()
            cache.append_frame(i, [(key, key), (key, key)], score_keys=[key, key])
        cache.close_segment([torch.ones(2, 4, 8)]*2)
        def run(qid):
            def done(hits):
                cache.record_activation(qid, hits)
                cache.release_snapshot(qid)
            return worker.answer(cache.acquire_snapshot(qid), 'question', done)
        expected = run(1)
        worker._set_window('encode')
        decoder(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=True)
        with native_decoder(decoder):
            decoder(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
        actual = run(2)
        self.assertEqual(actual['answer'], expected['answer'])
        self.assertEqual(actual['hit_frames'], expected['hit_frames'])
        self.assertFalse(cache.snapshots)


if __name__ == '__main__':
    unittest.main()
