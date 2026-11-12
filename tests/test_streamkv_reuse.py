"""Repeated-prefix checks independent of the optional StreamKV source checkout."""
import copy
from contextlib import ExitStack
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from transformers import LlavaOnevisionVideoProcessor

from model.streamkv.config import StreamKVConfig
from video_qa.streamkv_stream_vqa import StreamKVStreamVQA
from test_streamkv_runtime import cpu_cache_execution, tiny_runtime


class StreamKVPrefixReuseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(51)

    def sample(self, cutoffs):
        return dict(video_id='v', video_path='unused', conversations=[
            dict(question=('why', 'how', 'who')[i % 3], answer='answer', end_time=t)
            for i, t in enumerate(cutoffs)])

    def flow(self, runtime, sample, video, fps=1.):
        flow = StreamKVStreamVQA(anno=[sample], save_dir='/tmp', sample_fps=fps, qa_model=runtime)
        flow.load_video = lambda _: video
        return flow

    def test_effective_prefix_reuse_and_video_boundary(self):
        runtime = Mock()
        sample = self.sample((4, 8, 8, 8.9, 20, 2, 2.9, 8))
        video = np.zeros((8, 8, 8, 3), dtype=np.uint8)
        flow = self.flow(runtime, sample, video)
        flow.record_answer = Mock()
        flow.analyze_a_video(sample)
        self.assertEqual([len(c.args[0]) for c in runtime.encode_video.call_args_list], [4, 8, 2, 8])
        self.assertEqual(runtime.encode_init_prompt.call_count, 4)
        self.assertEqual(runtime.clear_cache.call_count, 5)  # rebuilds + final release
        self.assertEqual(flow.record_answer.call_count, 8)
        self.assertTrue(all(c.kwargs['max_new_tokens'] == 256 for c in flow.record_answer.call_args_list))
        # Even the same length in the next video must be encoded again.
        flow.analyze_a_video(self.sample((8, 8)))
        self.assertEqual(runtime.encode_video.call_count, 5)

    def test_sample_fps_rounding_and_failure_cleanup(self):
        runtime = Mock()
        sample = self.sample((8, 9, 10))
        flow = self.flow(runtime, sample, np.zeros((8, 8, 8, 3), dtype=np.uint8), fps=.5)
        flow.record_answer = Mock()
        flow.analyze_a_video(sample)
        self.assertEqual([len(c.args[0]) for c in runtime.encode_video.call_args_list], [4, 5])
        for failure in ('encode_video', 'record_answer'):
            with self.subTest(failure=failure):
                runtime = Mock()
                flow = self.flow(runtime, sample, np.zeros((8, 8, 8, 3), dtype=np.uint8), fps=.5)
                flow.record_answer = Mock()
                owner = flow if failure == 'record_answer' else runtime
                getattr(owner, failure).side_effect = RuntimeError('failed')
                with self.assertRaisesRegex(RuntimeError, 'failed'):
                    flow.analyze_a_video(self.sample((8, 8)))
                self.assertEqual(runtime.clear_cache.call_count, 2)
                getattr(owner, failure).side_effect = None
                flow.analyze_a_video(self.sample((8, 8)))
                self.assertEqual(runtime.encode_video.call_count, 2)

    @torch.inference_mode()
    def test_retrieval_or_decode_failure_discards_archive_before_retry(self):
        with cpu_cache_execution():
            for failed_call in (1, 2, 3):
                with self.subTest(failed_call=failed_call):
                    runtime = tiny_runtime(StreamKVConfig(n_local=24, retrieve_size=3,
                        retrieve_local_size=1, compression_ratio=.6, min_segment_size=2, max_segment_size=4))
                    runtime.model.config.vision_feature_select_strategy = 'full'
                    runtime.model.config.vision_feature_layer = -1
                    runtime.processor._processor.video_processor = LlavaOnevisionVideoProcessor(
                        size=dict(height=8, width=8))
                    sample = self.sample((4, 4))
                    flow = self.flow(runtime, sample, np.zeros((4, 8, 8, 3), dtype=np.uint8))
                    language_model = runtime.language_model
                    calls = 0
                    def fail_after_forward(*args, **kwargs):
                        nonlocal calls
                        output = language_model(*args, **kwargs)
                        calls += 1
                        if calls == failed_call:
                            raise RuntimeError('question failed')
                        return output
                    def record_answer(video_sample, question, **kwargs):
                        return runtime.question_answering(
                            dict(question=question['question'], prompt='answer'), max_new_tokens=2)
                    def failing_answer(*args, **kwargs):
                        with patch.object(runtime, 'language_model', side_effect=fail_after_forward):
                            return record_answer(*args, **kwargs)
                    flow.record_answer = failing_answer
                    with self.assertRaisesRegex(RuntimeError, 'question failed'):
                        flow.analyze_a_video(sample)
                    self.assertIsNone(runtime.kv_cache)
                    flow.record_answer = record_answer
                    with patch.object(runtime, 'encode_video', wraps=runtime.encode_video) as encode:
                        flow.analyze_a_video(sample)
                        self.assertEqual(encode.call_count, 1)
                    self.assertIsNone(runtime.kv_cache)

    @torch.inference_mode()
    def test_reuse_matches_fresh_encoding_retrieval_logits_and_answers(self):
        video = torch.randint(0, 256, (8, 8, 8, 3), dtype=torch.uint8).numpy()
        sample = self.sample((4, 8, 8, 8, 2, 2, 8))
        with cpu_cache_execution():
            for mode in ('uniform', 'semantic'):
                for ratio in (1., .6):
                    for summary in (False, True):
                        with self.subTest(mode=mode, ratio=ratio, summary=summary):
                            config = StreamKVConfig(n_local=24, retrieve_size=3, retrieve_local_size=1,
                                compression_ratio=ratio, compress_temp=1,
                                segment_mode=mode, min_segment_size=2, max_segment_size=4,
                                segment_theta=.99, use_segment_summary=summary)
                            results = []
                            for reuse in (False, True):
                                runtime = tiny_runtime(config)
                                runtime.model.config.vision_feature_select_strategy = 'full'
                                runtime.model.config.vision_feature_layer = -1
                                runtime.processor._processor.video_processor = LlavaOnevisionVideoProcessor(
                                    size=dict(height=8, width=8))
                                flow = self.flow(runtime, sample, video)
                                answers, traces, selections, archive_ids = [], [], [], []

                                def archive_tensors(layer):
                                    tensors = [layer.init_k, layer.init_v, layer.local_k, layer.local_v,
                                               *layer.global_remainder, layer.block_k[0].get_data()]
                                    tensors.extend(t for block in layer.global_blocks[0] for t in block.cpu_data)
                                    return tensors

                                def record_answer(video_sample, question, **kwargs):
                                    layers = runtime.kv_cache
                                    archive_ids.append(tuple(id(layer) for layer in layers))
                                    before = [[t.clone() for t in archive_tensors(layer)] for layer in layers]
                                    logits, selected = [], []
                                    with ExitStack() as stack:
                                        handle = runtime.model.lm_head.register_forward_hook(
                                            lambda m, a, out: logits.append(out.clone()))
                                        stack.callback(handle.remove)
                                        for layer in layers:
                                            original = layer.get_retrieved_kv
                                            def retrieve(*args, _layer=layer, _original=original, **kw):
                                                result = _original(*args, **kw)
                                                selected.append(copy.deepcopy(_layer.retrieved_block_indices))
                                                return result
                                            stack.enter_context(patch.object(layer, 'get_retrieved_kv', side_effect=retrieve))
                                        answers.append(runtime.question_answering(dict(question=question['question'],
                                            prompt=runtime.get_prompt(question['question'])), max_new_tokens=3))
                                    traces.append(logits)
                                    selections.append(selected)
                                    for layer, tensors in zip(layers, before):
                                        self.assertFalse(layer.to_retrieve)
                                        self.assertFalse(layer.retrieval_prefill)
                                        self.assertIsNone(layer.retrieved_block_indices)
                                        current = archive_tensors(layer)
                                        self.assertEqual(len(current), len(tensors))
                                        for actual, expected in zip(current, tensors):
                                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

                                flow.record_answer = record_answer
                                with patch.object(runtime, '_get_video_features', wraps=runtime._get_video_features) as vision, \
                                        patch.object(runtime, 'encode_video', wraps=runtime.encode_video) as encode:
                                    if reuse:
                                        flow.analyze_a_video(sample)
                                    else:
                                        # Original evaluator: rebuild on EVERY question.
                                        for question in sample['conversations']:
                                            runtime.clear_cache()
                                            runtime.encode_init_prompt()
                                            runtime.encode_video(torch.from_numpy(video[:question['end_time']]))
                                            record_answer(sample, question)
                                        runtime.clear_cache()
                                    self.assertEqual(encode.call_count, 4 if reuse else 7)
                                    if mode == 'semantic':
                                        self.assertEqual(vision.call_count, 22 if reuse else 40)
                                    else:
                                        self.assertEqual(vision.call_count, 6 if reuse else 11)
                                self.assertIsNone(runtime.kv_cache)
                                if reuse:
                                    self.assertEqual(archive_ids[1], archive_ids[2])
                                    self.assertEqual(archive_ids[2], archive_ids[3])
                                    self.assertEqual(archive_ids[4], archive_ids[5])
                                results.append((answers, traces, selections))
                            self.assertEqual(results[0][0], results[1][0])
                            self.assertEqual(results[0][2], results[1][2])
                            for fresh, reused in zip(results[0][1], results[1][1]):
                                self.assertEqual(len(fresh), len(reused))
                                for expected, actual in zip(fresh, reused):
                                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
