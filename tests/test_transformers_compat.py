"""Offline regression checks for the Transformers 5 ReKV inference paths.

Run: conda run -n rekv python -m unittest discover -s tests -v
CUDA checks exercise the real cache manager; CPU checks need no model downloads.
"""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from transformers import (
    LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM,
    Qwen2Config, Qwen2ForCausalLM, LlavaOnevisionConfig, VideoLlavaConfig,
    CLIPVisionModel, CLIPVisionConfig, CLIPImageProcessor,
    LlavaOnevisionVideoProcessor, VideoLlavaVideoProcessor,
)

from model.patch import patch_hf
from model.attention.kv_cache_manager import ContextManager
from model.llava_onevision_rekv import LlavaOneVision_ReKV
from model.video_llava_rekv import VideoLlava_ReKV
from model.longva_rekv import LongVA_ReKV
from longva.model import (
    LlavaConfig, LlavaLlamaForCausalLM, LlavaMistralConfig,
    LlavaMistralForCausalLM, LlavaQwenConfig, LlavaQwenForCausalLM,
)
from longva.model.multimodal_resampler.qformer import BertConfig, BertLMHeadModel


def tiny_config(config_class, **kwargs):
    return config_class(
        vocab_size=32, hidden_size=64, intermediate_size=96,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        pad_token_id=0, bos_token_id=1, eos_token_id=31,
        **kwargs,
    )


def patch(model, **kwargs):
    options = dict(n_init=2, n_local=32, topk=2, chunk_size=1,
                   block_size=2, max_cached_block=4, exc_block_size=2,
                   fattn=False, async_global_stream=False)
    options.update(kwargs)
    return patch_hf(model, **options)


def empty_cache(model, batch=1):
    decoder = getattr(model, 'model', model)
    return tuple((torch.empty(batch, 1, 0, 32, device=model.device),
                  torch.empty(batch, 1, 0, 32, device=model.device))
                 for _ in decoder.layers)


class TransformersCompatibilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    @torch.no_grad()
    def test_native_logits_and_incremental_cache(self):
        # With no eviction, ReKV must match native causal attention numerically.
        for config_class, model_class in [
            (LlamaConfig, LlamaForCausalLM), (MistralConfig, MistralForCausalLM),
            (Qwen2Config, Qwen2ForCausalLM),
        ]:
            with self.subTest(model=model_class.__name__):
                native = model_class(tiny_config(config_class, rope_theta=1000000.)).eval()
                model = patch(copy.deepcopy(native))
                ids = torch.tensor([[1, 2, 3, 4], [1, 5, 6, 7]])
                expected = native(ids, use_cache=False).logits
                output = model(ids, past_key_values=empty_cache(model, 2), use_cache=True,
                               output_hidden_states=True)
                torch.testing.assert_close(output.logits, expected, atol=2e-6, rtol=2e-5)
                self.assertEqual(len(output.hidden_states), 3)
                first = model(ids[:, :3], past_key_values=empty_cache(model, 2), use_cache=True)
                second = model(ids[:, 3:], past_key_values=first.past_key_values, use_cache=True)
                torch.testing.assert_close(second.logits, expected[:, -1:], atol=2e-6, rtol=2e-5)
                self.assertEqual(second.past_key_values[0][0].shape, (2, 1, 4, 32))
                # Bare decoders are now used by the HF multimodal models.
                decoder = patch(copy.deepcopy(native.model))
                result = decoder(ids, past_key_values=empty_cache(decoder, 2), use_cache=True,
                                 return_dict=False)
                self.assertEqual(result[0].shape, (2, 4, 64))

    @torch.no_grad()
    def test_longva_forward_generation_and_reload(self):
        for config_class, model_class in [
            (LlavaConfig, LlavaLlamaForCausalLM),
            (LlavaQwenConfig, LlavaQwenForCausalLM),
            (LlavaMistralConfig, LlavaMistralForCausalLM),
        ]:
            with self.subTest(model=model_class.__name__):
                model = model_class(tiny_config(config_class)).eval()
                ids = torch.tensor([[1, 2, 3]])
                self.assertEqual(model(ids, logits_to_keep=1).logits.shape, (1, 1, 32))
                self.assertEqual(model.generate(ids, max_new_tokens=2, min_new_tokens=2).shape[-1], 2)
                with tempfile.TemporaryDirectory() as directory:
                    model.save_pretrained(directory)
                    restored = model_class.from_pretrained(directory, local_files_only=True).eval()
                    torch.testing.assert_close(restored(ids).logits, model(ids).logits)
                patch(model)
                self.assertEqual(model(ids, past_key_values=empty_cache(model), use_cache=True).logits.shape,
                                 (1, 3, 32))

    @torch.no_grad()
    def test_multimodal_video_features_and_reload(self):
        for model_class, config_class, vision_type, text_type, tokens_per_frame in [
            (LlavaOneVision_ReKV, LlavaOnevisionConfig, 'siglip_vision_model', 'qwen2', 4),
            (VideoLlava_ReKV, VideoLlavaConfig, 'clip_vision_model', 'llama', 17),
        ]:
            with self.subTest(model=model_class.__name__):
                config = config_class(
                    text_config=tiny_config(Qwen2Config if text_type == 'qwen2' else LlamaConfig),
                    vision_config=dict(model_type=vision_type, hidden_size=32, intermediate_size=64,
                                       num_hidden_layers=2, num_attention_heads=4, image_size=8, patch_size=2),
                    vision_feature_layer=-1, vision_feature_select_strategy='full',
                )
                processor_class = LlavaOnevisionVideoProcessor if text_type == 'qwen2' else VideoLlavaVideoProcessor
                video_processor = processor_class(size={'shortest_edge': 8}, crop_size={'height': 8, 'width': 8})
                options = dict(processor=SimpleNamespace(video_processor=video_processor), n_frame_tokens=tokens_per_frame,
                               init_prompt_ids=[1, 2], n_local=64, topk=2, chunk_size=1)
                model = model_class(config, **options).eval()
                video = model.processor.video_processor(
                    videos=torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
                    do_sample_frames=False, return_tensors='pt',
                ).pixel_values_videos
                self.assertEqual(video.shape, (1, 2, 3, 8, 8))
                features = model._get_video_features(video)
                self.assertEqual(features.shape, (1, 2 * tokens_per_frame, 64))
                with tempfile.TemporaryDirectory() as directory:
                    model.save_pretrained(directory)
                    restored = model_class.from_pretrained(directory, local_files_only=True, **options).eval()
                    torch.testing.assert_close(restored._get_video_features(video), features)
                    # Existing v4 checkpoints store a full causal LM under
                    # language_model and keep the vision modules at the root.
                    legacy_state = {}
                    for key, value in model.state_dict().items():
                        if key.startswith('model.language_model.'):
                            key = key.replace('model.language_model.', 'language_model.model.', 1)
                        elif key.startswith('lm_head.'):
                            key = 'language_model.' + key
                        elif key.startswith('model.'):
                            key = key[len('model.'):]
                        legacy_state[key] = value
                    save_file(legacy_state, str(Path(directory) / 'model.safetensors'), metadata={'format': 'pt'})
                    restored, info = model_class.from_pretrained(
                        directory, local_files_only=True, output_loading_info=True, **options,
                    )
                    self.assertFalse(info['missing_keys'])
                    self.assertFalse(info['unexpected_keys'])
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[key], value)
                patch(model.language_model, n_local=64)
                output = model.language_model(inputs_embeds=features,
                                              past_key_values=empty_cache(model.language_model), use_cache=True)
                self.assertEqual(model.lm_head(output.last_hidden_state).shape,
                                 (1, 2 * tokens_per_frame, 32))

    @torch.no_grad()
    def test_longva_vision_checkpoint_loading(self):
        # Nested from_pretrained runs inside the outer loader's meta context.
        with tempfile.TemporaryDirectory() as directory:
            vision_path = Path(directory) / 'clip'
            model_path = Path(directory) / 'longva'
            CLIPVisionModel(CLIPVisionConfig(
                hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, image_size=8, patch_size=2,
            )).save_pretrained(vision_path)
            CLIPImageProcessor(size={'shortest_edge': 8}, crop_size={'height': 8, 'width': 8}).save_pretrained(vision_path)
            config = tiny_config(
                LlavaQwenConfig, mm_vision_tower=str(vision_path), mm_vision_select_layer=-1,
                mm_hidden_size=32, mm_projector_type='linear',
                mm_spatial_pool_mode='average', mm_spatial_pool_stride=2,
            )
            options = dict(n_frame_tokens=4, init_prompt_ids=[1, 2], n_local=32, topk=2, chunk_size=1)
            model = LongVA_ReKV(config, **options).eval()
            video = torch.randn(2, 3, 8, 8)
            expected = model._get_video_features(video)
            model.save_pretrained(model_path)
            restored = LongVA_ReKV.from_pretrained(model_path, local_files_only=True, **options).eval()
            torch.testing.assert_close(restored._get_video_features(video), expected)
            patch(restored)
            output = restored.model(inputs_embeds=expected, past_key_values=empty_cache(restored), use_cache=True)
            self.assertEqual(output.last_hidden_state.shape, (1, 8, 64))

    @torch.no_grad()
    def test_qformer_cross_attention_and_pruning(self):
        config = BertConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4, encoder_width=32,
                            add_cross_attention=True, cross_attention_freq=1, query_length=2)
        model = BertLMHeadModel(config).eval()
        args = dict(query_embeds=torch.randn(1, 2, 32), encoder_hidden_states=torch.randn(1, 4, 32),
                    encoder_attention_mask=torch.ones(1, 4), head_mask=torch.ones(4))
        self.assertEqual(model.bert(**args).last_hidden_state.shape, (1, 2, 32))
        for beams in [1, 2]:
            generated = model.generate(
                torch.tensor([[1, 2, 3]]), query_embeds=args['query_embeds'],
                encoder_hidden_states=args['encoder_hidden_states'],
                max_new_tokens=2, min_new_tokens=2, num_beams=beams, pad_token_id=0,
            )
            self.assertEqual(generated.shape, (1, 5))
        model.bert.encoder.layer[0].attention.prune_heads([1])
        model.bert.encoder.layer[0].attention.prune_heads([1, 2])
        args.pop('head_mask')
        self.assertEqual(model.bert(**args).last_hidden_state.shape, (1, 2, 32))

    @unittest.skipUnless(torch.cuda.is_available(), 'ReKV ContextManager requires CUDA')
    @torch.no_grad()
    def test_cuda_encoding_retrieval_and_decoding(self):
        for config_class, model_class in [
            (LlamaConfig, LlamaForCausalLM), (MistralConfig, MistralForCausalLM),
            (Qwen2Config, Qwen2ForCausalLM),
        ]:
            for fattn in [False, True]:
                with self.subTest(model=model_class.__name__, fattn=fattn):
                    model = patch(model_class(tiny_config(config_class)).cuda().eval(),
                                  n_local=8, fattn=fattn)
                    output = model(torch.tensor([[1, 2]], device='cuda'), use_cache=True)
                    caches = output.past_key_values
                    for _ in range(6):
                        output = model(torch.tensor([[3, 4]], device='cuda'), past_key_values=caches, use_cache=True)
                        self.assertIs(output.past_key_values[0], caches[0])
                    for cache in caches:
                        self.assertIsInstance(cache, ContextManager)
                        cache.set_retrieval()
                    retrieved = model(torch.tensor([[5, 6]], device='cuda'), past_key_values=caches, use_cache=True)
                    for cache in caches:
                        cache.reset_retrieval()
                    self.assertIsInstance(retrieved.past_key_values[0], tuple)
                    decoded = model(torch.tensor([[7]], device='cuda'),
                                    past_key_values=retrieved.past_key_values, use_cache=True)
                    self.assertEqual(decoded.logits.shape, (1, 1, 32))
                    self.assertTrue(torch.isfinite(decoded.logits).all())


if __name__ == '__main__':
    unittest.main()
