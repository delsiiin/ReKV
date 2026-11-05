"""Tiny real Qwen2.5-VL models: no checkpoint downloads required."""
import copy
from dataclasses import asdict
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2VLVideoProcessor

from model.rekv.qwen2_5_vl_vision import frame_features
from model.rekv.patch import patch_hf
from model.rekv.qwen2_5_vl_rekv import Qwen2_5_VL_ReKV
from model.suprkv.onevision import native_decoder
from model.suprkv.qwen2_5_vl import Qwen2_5_VLEngine, Qwen2_5_VLSummarizer
from model.suprkv.runtime import StreamingConfig, StreamingRuntime
from video_qa import base, run_eval
from video_qa.methods.rekv import ReKVMethod
from video_qa.methods.suprkv import make_config
from video_qa.model_paths import MODEL_PATHS
from model.video_backbone import model_family


class Tokenizer:
    eos_token_id = 63

    def __init__(self):
        self.texts = []

    def __call__(self, text, return_tensors=None, **kwargs):
        self.texts.append(text)
        ids = [1, 2, 3]
        return SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors else ids)

    def decode(self, ids, **kwargs):
        return 'a person walks'


def tiny_config():
    return Qwen2_5_VLConfig(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            eos_token_id=63, max_position_embeddings=4096,
            rope_parameters=dict(rope_type='default', rope_theta=1000000., mrope_section=[1, 1, 2])),
        vision_config=dict(depth=2, hidden_size=16, intermediate_size=32, num_heads=2,
            patch_size=2, spatial_merge_size=2, temporal_patch_size=2, window_size=8,
            out_hidden_size=32, fullatt_block_indexes=[1]),
        image_token_id=60, video_token_id=61, vision_start_token_id=58, vision_end_token_id=59)


def processor():
    return SimpleNamespace(tokenizer=Tokenizer(), video_processor=Qwen2VLVideoProcessor(
        patch_size=2, temporal_patch_size=2, merge_size=2))


class QwenSupportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.frames = torch.randint(0, 256, (3, 32, 48, 3), dtype=torch.uint8)

    @torch.inference_mode()
    def test_frame_blocks_are_independent_of_chunking_and_aspect_ratio(self):
        model = Qwen2_5_VLForConditionalGeneration(tiny_config()).eval()
        proc = processor()
        batched = frame_features(model, proc, self.frames)
        separate = torch.cat([frame_features(model, proc, frame[None]) for frame in self.frames])
        self.assertEqual(batched.shape, (3, 196, 32))
        torch.testing.assert_close(batched, separate)
        portrait = self.frames.transpose(1, 2)
        self.assertEqual(frame_features(model, proc, portrait).shape, batched.shape)
        with self.assertRaisesRegex(ValueError, 'nonempty RGB'):
            frame_features(model, proc, self.frames[:0])

    @torch.inference_mode()
    def test_rekv_decoder_matches_native_text_and_incremental_generation(self):
        native = Qwen2_5_VLForConditionalGeneration(tiny_config()).eval().model.language_model
        decoder = copy.deepcopy(native)
        patch_hf(decoder, n_init=3, n_local=32, topk=2, chunk_size=1,
                 block_size=2, max_cached_block=4, exc_block_size=2, fattn=False)
        cache = tuple((torch.empty(1, 2, 0, 8), torch.empty(1, 2, 0, 8)) for _ in decoder.layers)
        ids = torch.tensor([[1, 2, 3, 4]])
        expected = native(ids, use_cache=False).last_hidden_state
        output = decoder(ids[:, :3], past_key_values=cache, use_cache=True)
        output = decoder(ids[:, 3:], past_key_values=output.past_key_values, use_cache=True)
        torch.testing.assert_close(output.last_hidden_state, expected[:, -1:], atol=2e-6, rtol=2e-5)
        with native_decoder(decoder):
            torch.testing.assert_close(decoder(ids, use_cache=False).last_hidden_state, expected)

    @torch.inference_mode()
    def test_rekv_multimodal_reload_and_chunk_window(self):
        options = dict(processor=processor(), n_frame_tokens=196, init_prompt_ids=[1, 2, 3],
                       n_local=392, topk=2, chunk_size=1)
        model = Qwen2_5_VL_ReKV(tiny_config(), **options).eval()
        expected = frame_features(model, model.processor, self.frames)
        with tempfile.TemporaryDirectory() as path:
            model.save_pretrained(path)
            restored, info = Qwen2_5_VL_ReKV.from_pretrained(
                path, local_files_only=True, output_loading_info=True, **options)
            self.assertFalse(info['missing_keys'])
            self.assertFalse(info['unexpected_keys'])
            torch.testing.assert_close(frame_features(restored, restored.processor, self.frames), expected)
        # A caller's 64-frame chunk must be split before exceeding the local window.
        with patch.object(model, '_encode_video_chunk') as encode, \
                patch.object(model, 'calc_memory_usage', return_value=0):
            model.encode_video(self.frames, encode_chunk_size=64)
        self.assertEqual([call.args[0].shape[0] for call in encode.call_args_list], [2, 1])
        self.assertTrue(model.get_prompt('which?', mc=True).startswith('<|vision_end|>'))
        self.assertTrue(model.get_prompt('which?', mc=True).endswith('Best option: ('))

    @torch.inference_mode()
    def test_native_summary_preserves_spatial_positions_and_decode_offset(self):
        model = Qwen2_5_VLForConditionalGeneration(tiny_config()).eval()
        proc = processor()
        summary = Qwen2_5_VLSummarizer(model, proc, 'tiny')
        features = list(frame_features(model, proc, self.frames).split(1))
        frames = [dict(index=i, timestamp=float(i)) for i in range(3)]
        inputs, positions = summary._inputs_and_positions(features, frames)
        self.assertEqual(inputs.shape[1], summary.context_length(features, frames))
        self.assertFalse(torch.equal(positions[0], positions[1]))
        self.assertFalse(torch.equal(positions[1], positions[2]))
        calls = []
        def record(module, args, kwargs):
            calls.append(kwargs['position_ids'].clone())
        handle = model.model.language_model.register_forward_pre_hook(record, with_kwargs=True)
        try:
            # Keep generation going so the continuation position is exercised.
            proc.tokenizer.eos_token_id = -1
            summary.generate(features, frames, 3, lambda: None)
        finally:
            handle.remove()
        torch.testing.assert_close(calls[0], positions)
        self.assertTrue(torch.all(calls[1] == positions.max() + 1))
        self.assertTrue(torch.all(calls[2] == positions.max() + 2))

    @torch.inference_mode()
    def test_suprkv_summary_and_frame_query_compress_answer_resume_reset(self):
        for strategy in ('summary', 'frame_mean_query'):
            for granularity in ('layer', 'head'):
                with self.subTest(strategy=strategy, granularity=granularity):
                    config = StreamingConfig(model_family='qwen2_5_vl', device='cpu',
                        budget_tokens=1200, recent_frames=1, topk=2, group_size=8,
                        min_segment_frames=2, max_segment_frames=2,
                        summary_max_new_tokens=2, max_new_tokens=2,
                        scoring_strategy=strategy, compression_granularity=granularity)
                    model = Qwen2_5_VLForConditionalGeneration(tiny_config()).eval()
                    proc = processor()
                    engine = Qwen2_5_VLEngine(asdict(config), model=model, processor=proc)
                    with StreamingRuntime(config, lambda _: engine) as runtime:
                        for i in range(2):
                            runtime.append_frame(self.frames[i].numpy(), float(i))
                        stats = runtime.close_segment()
                        self.assertEqual(stats['summary_failures'], 0)
                        self.assertEqual(engine.cache.next_id, 2)
                        self.assertLessEqual(stats['persistent_bytes'], stats['budget_bytes'])
                        self.assertEqual(runtime.ask('what happens?')['answer'], 'a person walks')
                        self.assertIn(runtime.get_prompt('what happens?'), proc.tokenizer.texts)
                        self.assertFalse(engine.cache.snapshots)
                        runtime.append_frame(self.frames[2].numpy(), 2.)
                        runtime.close_segment()
                        self.assertEqual(engine.cache.next_id, 3)
                        runtime.reset()
                        self.assertEqual(engine.cache.next_id, 0)
                        self.assertEqual(runtime.memory_stats()['retained_frames'], 0)

    def test_cli_routes_presets_and_custom_paths_to_both_backbones(self):
        for name in (name for name in MODEL_PATHS if model_family(name) == 'qwen2_5_vl'):
            for method in ('rekv', 'suprkv'):
                args = run_eval.build_parser().parse_args([
                    '--method', method, '--model', name, '--dataset', 'rvs_ego',
                    '--model_path', '/local/qwen'])
                if method == 'suprkv':
                    config = make_config(args)
                    self.assertEqual(config.model_family, 'qwen2_5_vl')
                    self.assertEqual(config.model_path, '/local/qwen')
                else:
                    with patch('video_qa.methods.rekv.import_module') as imported:
                        ReKVMethod().load(args)
                    imported.assert_called_once_with('model.rekv.qwen2_5_vl_rekv')
                    self.assertEqual(imported.return_value.load_model.call_args.kwargs['model_path'], '/local/qwen')
                    self.assertIn('--model_path', ReKVMethod().worker_arguments(args))
        args = base.build_parser('rekv').parse_args([
            '--model', 'qwen2_5_vl_7b', '--n_local', '100',
            '--anno_path', 'unused', '--save_dir', 'unused'])
        with self.assertRaisesRegex(ValueError, 'n_local >= 196'):
            ReKVMethod().validate(args)

    @unittest.skipUnless(torch.cuda.is_available(), 'ReKV ContextManager requires CUDA')
    @torch.inference_mode()
    def test_rekv_real_cache_encode_retrieve_and_resume(self):
        model = Qwen2_5_VL_ReKV(tiny_config(), processor=processor(), n_frame_tokens=196,
            init_prompt_ids=[1, 2, 3], n_local=392, topk=1, chunk_size=1).cuda().eval()
        patch_hf(model.language_model, n_init=3, n_local=392, topk=1, chunk_size=1,
                 block_size=196, max_cached_block=4, exc_block_size=196, fattn=False,
                 async_global_stream=False)
        model.encode_init_prompt()
        model.encode_video(self.frames)
        prompt = dict(question='what?', prompt=model.get_prompt('what?'))
        for indices in (None, [[0]]):
            answer = model.question_answering(prompt, max_new_tokens=2, retrieved_indices=indices)
            self.assertEqual(answer, 'a person walks')
        model.encode_video(self.frames[:1])
        self.assertTrue(all(cache.length == 3 + 4 * 196 for cache in model.kv_cache))


if __name__ == '__main__':
    unittest.main()
