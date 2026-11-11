"""Real tiny OneVision/processor tests; no downloaded checkpoints or CUDA."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from transformers import (LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration,
                          LlavaOnevisionImageProcessor, LlavaOnevisionProcessor,
                          LlavaOnevisionVideoProcessor, Qwen2Config)

from model.infinipot.cache import process_kv_cache
from model.infinipot.config import InfiniPotConfig
from model.infinipot.onevision import InfiniPotOneVisionRuntime
from model.infinipot.video import frame_indices, load_video
from video_qa import base, run_eval
from video_qa.infinipot_stream_vqa import InfiniPotStreamVQA
from video_qa.methods import get_method, validate_arguments
from test_qwen2_5_vl_native import native_processor


def tiny_runtime(config=None, size=16):
    text = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=48,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=8192, eos_token_id=63, pad_token_id=62)
    model = LlavaOnevisionForConditionalGeneration(LlavaOnevisionConfig(
        text_config=text, video_token_index=61, image_token_index=60,
        vision_feature_layer=-1, vision_feature_select_strategy='full',
        vision_config=dict(model_type='siglip_vision_model', hidden_size=16, intermediate_size=24,
                           num_hidden_layers=2, num_attention_heads=2, image_size=size,
                           patch_size=2, vision_use_head=False))).eval()
    model.set_attn_implementation('eager')
    model.generation_config.do_sample = False
    processor = LlavaOnevisionProcessor(
        tokenizer=native_processor().tokenizer,
        image_processor=LlavaOnevisionImageProcessor(size=dict(height=size, width=size)),
        video_processor=LlavaOnevisionVideoProcessor(size=dict(height=size, width=size)),
        num_image_tokens=(size // 2) ** 2, vision_feature_select_strategy='full',
        video_token='<|video_pad|>', image_token='<|image_pad|>',
        chat_template=(
            "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' }}"
            "{% for item in message['content'] %}{% if item['type']=='video' %}"
            "{{ '<|video_pad|>' }}{% else %}{{ item['text'] }}{% endif %}{% endfor %}"
            "{{ '<|im_end|>\\n' }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"))
    return InfiniPotOneVisionRuntime(model, processor, config, max_new_tokens=3)


class OneVisionInfiniPotTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def video(self, frames):
        return torch.randint(0, 256, (frames, 3, 20, 24), dtype=torch.uint8)

    @torch.inference_mode()
    def test_uncompressed_block_path_matches_native_logits_and_newline(self):
        runtime = tiny_runtime()
        for frames in (1, 3):
            with self.subTest(frames=frames):
                runtime.set_video(self.video(frames))
                native = runtime.prepare_inputs('describe')
                expected = runtime.model(**native).logits[:, -1:]
                raw_ids = runtime.processor.tokenizer(runtime._prompt('describe'), return_tensors='pt').input_ids
                logits = []
                hook = runtime.model.lm_head.register_forward_hook(lambda m, a, out: logits.append(out.clone()))
                try:
                    actual_answer = runtime.block_process(raw_ids)
                finally:
                    hook.remove()
                torch.testing.assert_close(logits[0], expected, atol=1e-6, rtol=1e-5)
                self.assertEqual(actual_answer, runtime.generate(native))
                self.assertEqual(int((native.input_ids == 61).sum()), frames * 16 + 1)

    @torch.inference_mode()
    def test_compression_bounds_prefix_positions_and_tail(self):
        for method in ('infinipot-v', 'uniform', 'swa'):
            for pool in (False, True):
                for frames in (9, 10, 11, 13):
                    with self.subTest(method=method, pool=pool, frames=frames):
                        runtime = tiny_runtime(InfiniPotConfig(
                            block_size=8, compress_frame_num=6,
                            compression_method=method, adaptive_pooling=pool))
                        runtime.set_video(self.video(frames))
                        calls, compressions, encoded = [], [], []
                        original_encode = runtime._encode_frames
                        def encode(start, end):
                            encoded.append((start, end))
                            return original_encode(start, end)
                        def record(module, args, kwargs):
                            calls.append((kwargs['position_ids'].clone(),
                                          kwargs['past_key_values'].get_seq_length()))
                        def compress(cache, model, system, inst, tokens, keep, **kwargs):
                            before = [(l.keys.clone(), l.values.clone()) for l in cache.layers]
                            result = process_kv_cache(cache, model, system, inst, tokens, keep, **kwargs)
                            self.assertEqual(tokens, 16)
                            self.assertEqual(cache.get_seq_length(), system + keep * tokens)
                            for layer, (keys, values) in zip(cache.layers, before):
                                torch.testing.assert_close(layer.keys[:, :, :system], keys[:, :, :system])
                                torch.testing.assert_close(layer.values[:, :, :system], values[:, :, :system])
                                # Every retained K/V pair must come from the same original position.
                                for head in range(keys.shape[1]):
                                    distances = (layer.keys[0, head, system:, None] - keys[0, head, None, system:]).abs().amax(-1)
                                    self.assertTrue((distances.min(-1).values == 0).all())
                                    indices = distances.argmin(-1) + system
                                    torch.testing.assert_close(layer.values[0, head, system:], values[0, head, indices])
                            compressions.append(keep)
                            return result
                        hook = runtime.model.model.language_model.register_forward_pre_hook(record, with_kwargs=True)
                        try:
                            with patch.object(runtime, '_encode_frames', side_effect=encode), \
                                    patch('model.infinipot.onevision.process_kv_cache', side_effect=compress):
                                answer = runtime.question_answering({'prompt': 'describe'})
                        finally:
                            hook.remove()
                        self.assertIsInstance(answer, str)
                        self.assertEqual(encoded[0], (0, 8))
                        prefix = calls[0][0].shape[1] - 8 * 16
                        self.assertEqual(compressions[-1], 7 if frames % 2 else 6)
                        for (start, end), (positions, cached) in zip(encoded, calls):
                            self.assertLessEqual(cached + positions.numel(), prefix + 8 * 16)
                            self.assertEqual(int(positions[0, 0]), 0 if start == 0 else prefix + start * 16)
                        # Question starts at the original video-end position, even after eviction.
                        self.assertEqual(int(calls[len(encoded)][0][0, 0]), prefix + frames * 16)
                        self.assertEqual(calls[len(encoded)][1], prefix + 8 * 16)
                        self.assertFalse(hasattr(runtime.model.config, 'height_width'))

    def test_native_fallback_question_isolation_and_reset(self):
        runtime = tiny_runtime(InfiniPotConfig(block_size=8, compress_frame_num=6))
        for frames in (0, 1, 8, 11):
            with self.subTest(frames=frames):
                runtime.set_video(self.video(frames) if frames else None)
                with patch.object(runtime, 'generate', wraps=runtime.generate) as native:
                    first = runtime.question_answering({'prompt': 'describe'}, max_new_tokens=1)
                    runtime.question_answering({'prompt': 'token9'})
                    self.assertEqual(first, runtime.question_answering({'prompt': 'describe'}))
                self.assertEqual(native.call_count, 0 if frames > 8 else 3)
                runtime.reset()
                self.assertIsNone(runtime.video)
        runtime.close()

    @torch.inference_mode()
    def test_reused_prefill_matches_fresh_logits_without_reencoding(self):
        for method in ('infinipot-v', 'uniform', 'swa'):
            for pooling in (False, True):
                with self.subTest(method=method, pooling=pooling):
                    runtime = tiny_runtime(InfiniPotConfig(
                        block_size=8, compress_frame_num=6,
                        compression_method=method, adaptive_pooling=pooling))
                    video = self.video(13)
                    questions = ('token9', 'token10 token11 token12', 'token9')
                    logits = []
                    hook = runtime.model.lm_head.register_forward_hook(
                        lambda m, a, out: logits.append(out.clone()))
                    try:
                        expected = []
                        # Execute the previous path: generate directly from a
                        # newly computed prefill, without snapshotting/cloning.
                        with patch.object(runtime, '_get_video_prefill', side_effect=runtime._prefill_video):
                            for question in questions:
                                runtime.set_video(video)
                                logits.clear()
                                answer = runtime.question_answering({'prompt': question})
                                expected.append((answer, logits[:]))
                        runtime.set_video(video)
                        with patch.object(runtime, '_prefill_video', wraps=runtime._prefill_video) as prefill, \
                                patch.object(runtime, '_encode_frames', wraps=runtime._encode_frames) as encode, \
                                patch('model.infinipot.onevision.process_kv_cache', wraps=process_kv_cache) as compress:
                            snapshot = None
                            for question, (answer, expected_logits) in zip(questions, expected):
                                logits.clear()
                                self.assertEqual(runtime.question_answering({'prompt': question}), answer)
                                self.assertEqual(len(logits), len(expected_logits))
                                for actual, wanted in zip(logits, expected_logits):
                                    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                                self.assertEqual(prefill.call_count, 1)
                                self.assertEqual(encode.call_count, 4)
                                self.assertEqual(compress.call_count, 3)
                                cache = runtime._prefill_cache
                                if snapshot is None:
                                    snapshot = [(l.keys.clone(), l.values.clone()) for l in cache.layers]
                                for layer, (keys, values) in zip(cache.layers, snapshot):
                                    torch.testing.assert_close(layer.keys, keys, rtol=0, atol=0)
                                    torch.testing.assert_close(layer.values, values, rtol=0, atol=0)
                                self.assertEqual(cache.get_seq_length(),
                                                 runtime._prefill_prefix_ids.shape[1] + 8 * 16)
                    finally:
                        hook.remove()
                        runtime.close()

    @torch.inference_mode()
    def test_prefill_invalidation_for_video_prompt_config_and_lifecycle(self):
        runtime = tiny_runtime(InfiniPotConfig(block_size=8, compress_frame_num=6))
        video = self.video(13)
        runtime.set_video(video)
        ids = runtime.processor.tokenizer(runtime._prompt('token9'), return_tensors='pt').input_ids
        with patch.object(runtime, '_prefill_video', wraps=runtime._prefill_video) as prefill:
            runtime.block_process(ids)
            runtime.block_process(ids)
            self.assertEqual(prefill.call_count, 1)
            # Different text before the video must not hit a same-length cache.
            changed = ids.clone()
            changed[0, 0] = 7
            runtime.block_process(changed)
            self.assertEqual(prefill.call_count, 2)
            torch.testing.assert_close(runtime._prefill_prefix_ids, changed[:, :int(
                (changed[0] == 61).nonzero()[0])])
            runtime.config = InfiniPotConfig(block_size=8, compress_frame_num=5)
            runtime.block_process(changed)
            self.assertEqual(prefill.call_count, 3)
            for new_video in (video, self.video(13), self.video(11)):
                runtime.set_video(new_video)
                self.assertIsNone(runtime._prefill_cache)
                runtime.block_process(ids)
            self.assertEqual(prefill.call_count, 6)
        runtime.set_video(None)
        self.assertIsNone(runtime._prefill_cache)
        for clear in (runtime.reset, runtime.close):
            runtime.set_video(video)
            runtime.block_process(ids)
            clear()
            self.assertIsNone(runtime.video)
            self.assertIsNone(runtime._prefill_cache)
            self.assertIsNone(runtime._prefill_prefix_ids)

    @torch.inference_mode()
    def test_failed_generation_cannot_mutate_reusable_prefill(self):
        runtime = tiny_runtime(InfiniPotConfig(block_size=8, compress_frame_num=6))
        runtime.set_video(self.video(13))
        expected = runtime.question_answering({'prompt': 'token9'})
        snapshot = [(l.keys.clone(), l.values.clone()) for l in runtime._prefill_cache.layers]
        forward = runtime._forward
        def fail_after_cache_update(*args, **kwargs):
            forward(*args, **kwargs)
            raise RuntimeError('generation failed after cache update')
        with patch.object(runtime, '_forward', side_effect=fail_after_cache_update), \
                self.assertRaisesRegex(RuntimeError, 'generation failed'):
            runtime.question_answering({'prompt': 'token10 token11'})
        for layer, (keys, values) in zip(runtime._prefill_cache.layers, snapshot):
            torch.testing.assert_close(layer.keys, keys, rtol=0, atol=0)
            torch.testing.assert_close(layer.values, values, rtol=0, atol=0)
        with patch.object(runtime, '_prefill_video', wraps=runtime._prefill_video) as prefill:
            self.assertEqual(runtime.question_answering({'prompt': 'token9'}), expected)
            prefill.assert_not_called()
        # A failed rebuild must not leave a stale/partial snapshot available.
        runtime.set_video(self.video(11))
        with patch.object(runtime, '_prefill_video', side_effect=RuntimeError('prefill failed')), \
                self.assertRaisesRegex(RuntimeError, 'prefill failed'):
            runtime.question_answering({'prompt': 'token9'})
        self.assertIsNone(runtime._prefill_cache)
        runtime.question_answering({'prompt': 'token9'})
        self.assertIsNotNone(runtime._prefill_cache)

    def test_real_frame_token_count(self):
        runtime = tiny_runtime(size=56)
        runtime.set_video(self.video(2))
        with torch.inference_mode():
            self.assertEqual(runtime._encode_frames(0, 2).shape[1], 2 * 196)

    def test_sdpa_and_eager_agree_after_compression(self):
        runtime = tiny_runtime(InfiniPotConfig(block_size=8, compress_frame_num=6))
        runtime.set_video(self.video(13))
        logits = []
        hook = runtime.model.lm_head.register_forward_hook(lambda m, a, out: logits.append(out.clone()))
        try:
            eager_answer = runtime.question_answering({'prompt': 'describe'})
            eager_logits = logits[:]
            logits.clear()
            runtime.model.set_attn_implementation('sdpa')
            # Rebuild prefill as well when comparing attention implementations.
            runtime.set_video(runtime.video)
            self.assertEqual(runtime.question_answering({'prompt': 'describe'}), eager_answer)
            self.assertEqual(len(logits), len(eager_logits))
            for expected, actual in zip(eager_logits, logits):
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        finally:
            hook.remove()

    def test_rvs_actual_processor_and_visible_prefix_replay(self):
        runtime = tiny_runtime(InfiniPotConfig(block_size=8, compress_frame_num=6))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'video.npy'
            np.save(path, np.stack([np.full((20, 24, 3), i, dtype=np.uint8) for i in range(12)]))
            sample = dict(video_id='v', video_path=str(path), conversations=[
                dict(question='describe', answer='yes', end_time=end) for end in (0, 9, 9, 11, 11)])
            flow = InfiniPotStreamVQA(anno=[sample], save_dir=directory, sample_fps=2,
                                     qa_model=runtime, num_chunks=1, chunk_idx=0)
            with patch.object(runtime, 'set_video', wraps=runtime.set_video) as videos, \
                    patch.object(runtime, '_prefill_video', wraps=runtime._prefill_video) as prefill:
                flow.analyze()
            self.assertEqual(prefill.call_count, 2)
            observed = [call.args[0] for call in videos.call_args_list]
            self.assertIsNone(observed[0])
            self.assertEqual([len(video) for video in observed[1:]], [9, 11])
            self.assertEqual([int(video.max()) for video in observed[1:]], [8, 10])
            self.assertEqual(len(flow.records), 5)
            self.assertIsNone(runtime.video)
            self.assertIsNone(runtime._prefill_cache)

    def test_single_frames_odd_limits_and_visible_prefix_sampling(self):
        self.assertEqual(frame_indices(1, 1, frame_factor=1).tolist(), [0])
        self.assertEqual(len(frame_indices(11, 1, max_frames=5, frame_factor=1)), 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'video.npy'
            np.save(path, np.stack([np.full((7, 9, 3), i, dtype=np.uint8) for i in range(5)]))
            options = InfiniPotOneVisionRuntime.video_loading_options
            one = load_video(path, end_time=1, **options)
            self.assertEqual(tuple(one.shape), (1, 3, 7, 9))
            prefix = load_video(path, end_time=3, **options)
            self.assertEqual(prefix[:, 0, 0, 0].tolist(), [0, 1, 2])
            self.assertIsNone(load_video(path, end_time=0, **options))

    def test_onevision_presets_all_dataset_routes(self):
        method = get_method('infinipot')
        for model in ('llava_ov_0.5b', 'llava_ov_7b', 'llava_ov_72b'):
            for dataset, (_, mode, _) in run_eval.DATASETS.items():
                args = run_eval.build_parser().parse_args([
                    '--method', 'infinipot', '--model', model, '--dataset', dataset])
                validate_arguments(args, method)
                with patch.object(run_eval, 'load_samples', return_value=[{}]), \
                        patch.object(run_eval.multiprocessing, 'Process', return_value=Mock(exitcode=0)) as launch, \
                        patch.object(run_eval, 'exec'):
                    run_eval.evaluate(args, dataset)
                command = launch.call_args.kwargs['args'][0]
                self.assertEqual(command[2], f'video_qa.infinipot_{mode}_vqa')
                worker = base.build_parser('infinipot').parse_args(command[3:])
                validate_arguments(worker, method)
                self.assertEqual(worker.model, model)
                self.assertEqual(method.config(worker), InfiniPotConfig())


if __name__ == '__main__':
    unittest.main()
