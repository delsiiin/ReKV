"""Differential prefill-reuse tests with real tiny Qwen and OneVision models."""
import unittest
from unittest.mock import patch

import torch

from model.infinipot.config import InfiniPotConfig
from model.infinipot.core import InfiniPotCore
from test_infinipot import tiny_runtime as qwen_runtime
from test_infinipot_onevision import tiny_runtime as onevision_runtime


class PrefillReuseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def video(self, family, frames):
        return torch.randint(0, 256, (frames * (2 if family == 'qwen' else 1), 3, 8, 8),
                             dtype=torch.uint8)

    def snapshot(self, runtime):
        return [(layer.keys.clone(), layer.values.clone()) for layer in runtime._prefill_cache.layers]

    def assert_snapshot(self, runtime, before):
        for layer, (keys, values) in zip(runtime._prefill_cache.layers, before):
            torch.testing.assert_close(layer.keys, keys, rtol=0, atol=0)
            torch.testing.assert_close(layer.values, values, rtol=0, atol=0)

    @torch.inference_mode()
    def test_native_matches_uncached_generate_with_greedy_sampling_and_beams(self):
        for family, factory in (('qwen', qwen_runtime), ('onevision', onevision_runtime)):
            for mode in ('greedy', 'sample', 'multi_sample', 'beam'):
                for frames in (1, 8):
                    with self.subTest(family=family, mode=mode, frames=frames):
                        runtime = factory(InfiniPotConfig(block_size=8, compress_frame_num=6))
                        runtime.model.generation_config.repetition_penalty = 1.15
                        if mode in ('sample', 'multi_sample'):
                            runtime.model.generation_config.do_sample = True
                            runtime.model.generation_config.top_k = 10
                            if mode == 'multi_sample':
                                runtime.model.generation_config.num_return_sequences = 2
                        elif mode == 'beam':
                            runtime.model.generation_config.num_beams = 2
                        runtime.set_video(self.video(family, frames))
                        logits = []
                        hook = runtime.model.register_forward_hook(
                            lambda m, a, out: logits.append(out.logits[:, -1:].clone()))
                        questions = ('token9', 'token10 token11 token12', 'token9')
                        expected = []
                        try:
                            for question in questions:
                                inputs = dict(runtime.prepare_inputs(question))
                                inputs.pop('second_per_grid_ts', None)
                                if family == 'qwen':
                                    runtime.model.model.rope_deltas = None
                                logits.clear()
                                torch.manual_seed(93)
                                ids = runtime.model.generate(**inputs, max_new_tokens=runtime.MAX_GEN_TOKENS)
                                answer = runtime.processor.batch_decode(
                                    [ids[0, inputs['input_ids'].shape[1]:]],
                                    skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                                expected.append((answer, logits[:]))
                            before = None
                            with patch.object(runtime.model.model, 'get_video_features',
                                              wraps=runtime.model.model.get_video_features) as vision:
                                for question, (answer, wanted) in zip(questions, expected):
                                    logits.clear()
                                    torch.manual_seed(93)
                                    self.assertEqual(runtime.question_answering({'prompt': question}), answer)
                                    self.assertEqual(len(logits), len(wanted))
                                    for actual, reference in zip(logits, wanted):
                                        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
                                    self.assertEqual(vision.call_count, 1)
                                    self.assertEqual(runtime._prefill_mode, 'native')
                                    if before is None:
                                        before = self.snapshot(runtime)
                                    self.assert_snapshot(runtime, before)
                                    if family == 'qwen':
                                        self.assertIsNone(runtime.model.model.rope_deltas)
                        finally:
                            hook.remove()
                            runtime.close()

    @torch.inference_mode()
    def test_qwen_block_reuse_preserves_question_specific_generation_positions(self):
        for method in ('infinipot-v', 'uniform', 'swa'):
            for pooling in (False, True):
                with self.subTest(method=method, pooling=pooling):
                    runtime = qwen_runtime(InfiniPotConfig(block_size=8, compress_frame_num=6,
                        compression_method=method, adaptive_pooling=pooling))
                    video = self.video('qwen', 13)
                    runtime.set_video(video)
                    # The old prefill implementation is the baseline, including
                    # its unusual generation start AFTER the full prompt.
                    old_prefill = InfiniPotCore._block_wise_prefill.__get__(runtime)
                    questions = ('token9', 'token10 token11 token12', 'token9')
                    records = []
                    generation = runtime._generation_stage
                    def record_generation(inputs, cache, positions):
                        calls = []
                        def record(m, args, kwargs, out):
                            calls.append((kwargs['position_ids'].clone(), out.logits[:, -1:].clone()))
                        hook = runtime.model.register_forward_hook(record, with_kwargs=True)
                        try:
                            result = generation(inputs, cache, positions)
                        finally:
                            hook.remove()
                        records.append(calls)
                        return result
                    with patch.object(runtime, '_generation_stage', side_effect=record_generation):
                        with patch.object(runtime, '_block_wise_prefill', side_effect=old_prefill):
                            answers = [runtime.question_answering({'prompt': q}) for q in questions]
                        expected = records[:]
                        records.clear()
                        with patch.object(runtime, '_visual_encoding_block',
                                          wraps=runtime._visual_encoding_block) as vision:
                            before = None
                            for question, answer in zip(questions, answers):
                                self.assertEqual(runtime.question_answering({'prompt': question}), answer)
                                self.assertEqual(vision.call_count, 4)
                                if before is None:
                                    before = self.snapshot(runtime)
                                self.assert_snapshot(runtime, before)
                        for actual, wanted in zip(records, expected):
                            self.assertEqual(len(actual), len(wanted))
                            for (pos, logits), (ref_pos, ref_logits) in zip(actual, wanted):
                                torch.testing.assert_close(pos, ref_pos, rtol=0, atol=0)
                                torch.testing.assert_close(logits, ref_logits, rtol=0, atol=0)
                        self.assertNotEqual(int(records[0][0][0][0, 0, 0]),
                                            int(records[1][0][0][0, 0, 0]))
                    runtime.close()

    @torch.inference_mode()
    def test_native_disabled_compression_invalidation_and_failure_isolation(self):
        for family, factory in (('qwen', qwen_runtime), ('onevision', onevision_runtime)):
            with self.subTest(family=family):
                runtime = factory(InfiniPotConfig(block_size=-1, compress_frame_num=0))
                video = self.video(family, 11)
                runtime.set_video(video)
                expected = runtime.question_answering({'prompt': 'token9'})
                before = self.snapshot(runtime)
                forward = runtime.model.forward
                def fail_after_forward(*args, **kwargs):
                    forward(*args, **kwargs)
                    raise RuntimeError('generation failed')
                with patch.object(runtime.model, 'forward', side_effect=fail_after_forward), \
                        self.assertRaisesRegex(RuntimeError, 'generation failed'):
                    runtime.question_answering({'prompt': 'token10 token11'})
                self.assert_snapshot(runtime, before)
                with patch.object(runtime.model.model, 'get_video_features',
                                  wraps=runtime.model.model.get_video_features) as vision:
                    self.assertEqual(runtime.question_answering({'prompt': 'token9'}), expected)
                    self.assertEqual(vision.call_count, 0)
                    runtime.set_video(video)
                    runtime.question_answering({'prompt': 'token9'})
                    self.assertEqual(vision.call_count, 1)
                    runtime.processor.chat_template = 'token7 ' + runtime.processor.chat_template
                    runtime.question_answering({'prompt': 'token9'})
                    self.assertEqual(vision.call_count, 2)
                    runtime.set_video(self.video(family, 3))
                    runtime.question_answering({'prompt': 'token9'})
                    self.assertEqual(vision.call_count, 3)
                runtime.reset()
                self.assertIsNone(runtime._prefill_cache)
                runtime.question_answering({'prompt': 'text only'})
                self.assertIsNone(runtime._prefill_cache)
                runtime.close()

    @torch.inference_mode()
    def test_switching_block_native_and_text_keeps_only_current_snapshot(self):
        for family, factory in (('qwen', qwen_runtime), ('onevision', onevision_runtime)):
            with self.subTest(family=family):
                runtime = factory(InfiniPotConfig(block_size=8, compress_frame_num=6))
                video = self.video(family, 11)
                runtime.set_video(video)
                expected = runtime.question_answering({'prompt': 'token9'})
                self.assertEqual(runtime._prefill_mode, 'block')
                runtime.generate(runtime.prepare_inputs('token9'))
                self.assertEqual(runtime._prefill_mode, 'native')
                self.assertEqual(runtime.question_answering({'prompt': 'token9'}), expected)
                self.assertEqual(runtime._prefill_mode, 'block')
                runtime.processor.chat_template = 'token7 ' + runtime.processor.chat_template
                with patch.object(runtime.model.model, 'get_video_features',
                                  wraps=runtime.model.model.get_video_features) as vision:
                    runtime.question_answering({'prompt': 'token9'})
                    self.assertGreater(vision.call_count, 0)
                runtime.set_video(video)
                with patch.object(runtime.model.model, 'get_video_features', side_effect=RuntimeError('prefill failed')), \
                        self.assertRaisesRegex(RuntimeError, 'prefill failed'):
                    runtime.question_answering({'prompt': 'token9'})
                self.assertIsNone(runtime._prefill_cache)
                runtime.question_answering({'prompt': 'token9'})
                before = self.snapshot(runtime)
                method = '_generation_stage' if family == 'qwen' else '_forward'
                with patch.object(runtime, method, side_effect=RuntimeError('generation failed')), \
                        self.assertRaisesRegex(RuntimeError, 'generation failed'):
                    runtime.question_answering({'prompt': 'token10 token11'})
                self.assert_snapshot(runtime, before)
                runtime.set_video(None)
                runtime.question_answering({'prompt': 'text only'})
                self.assertIsNone(runtime._prefill_cache)
                runtime.close()

    @torch.inference_mode()
    def test_explicit_no_cache_preserves_native_generation_and_eos(self):
        for family, factory in (('qwen', qwen_runtime), ('onevision', onevision_runtime)):
            with self.subTest(family=family):
                runtime = factory()
                runtime.set_video(self.video(family, 1))
                generation = runtime.model.generation_config
                generation.use_cache = False
                generation.forced_eos_token_id = generation.eos_token_id
                runtime.MAX_GEN_TOKENS = 1
                inputs = dict(runtime.prepare_inputs('token9'))
                inputs.pop('second_per_grid_ts', None)
                expected = runtime.model.generate(**inputs, max_new_tokens=1)
                with patch.object(runtime.model, 'generate', wraps=runtime.model.generate) as native:
                    runtime.question_answering({'prompt': 'token9'})
                self.assertNotIn('past_key_values', native.call_args.kwargs)
                self.assertIsNone(runtime._prefill_cache)
                generation.use_cache = True
                actual = []
                generate = runtime.model.generate
                def record_generate(**kwargs):
                    result = generate(**kwargs)
                    actual.append(result)
                    return result
                with patch.object(runtime.model, 'generate', side_effect=record_generate):
                    runtime.question_answering({'prompt': 'token9'})
                torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
                self.assertEqual(int(actual[0][0, -1]), generation.eos_token_id)
                runtime.close()


if __name__ == '__main__':
    unittest.main()
