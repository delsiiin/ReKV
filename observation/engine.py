"""Uncompressed causal prefill of local LLaVA-OneVision with a window observer."""
import importlib.metadata
import math
from pathlib import Path
import time

import numpy as np
import torch

from observation.metrics import WINDOW_SIZE, build_views, window_importance
from observation.spatial import video_token_layout
from observation.kv import save_kv_snapshot


INIT_PROMPT = '<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user '
ATTENTION_NAME = 'rvs_window_observer'


class Cancelled(Exception):
    pass


@torch.inference_mode()
def greedy_token_ids(decoder, lm_head, output, max_new_tokens, stop_token_ids,
                     check_cancel=lambda: None, on_token=lambda count: None):
    """Decode from an existing native prefill cache; never call the observer here."""
    if max_new_tokens < 1 or output.past_key_values is None:
        raise ValueError('Generation requires a prefill cache and a positive token budget')
    generated = []
    for step in range(max_new_tokens):
        check_cancel()
        last_hidden = output.last_hidden_state[:, -1:]
        token = int(lm_head(last_hidden)[0, -1].argmax())
        generated.append(token)
        on_token(len(generated))
        if token in stop_token_ids:
            return generated, 'eos'
        if step + 1 < max_new_tokens:
            output = decoder(input_ids=torch.tensor([[token]], device=last_hidden.device),
                             past_key_values=output.past_key_values, use_cache=True, return_dict=True)
    return generated, 'max_new_tokens'


def install_observer(decoder, callback):
    """Dispatch post-RoPE Q/K to the observer, preserving native SDPA outputs."""
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import AttentionMaskInterface

    def attention(module, query, key, value, attention_mask, **kwargs):
        if kwargs.get('sliding_window') is not None:
            raise ValueError("This experiment requires full causal attention")
        module._rvs_observer(module.layer_idx, window_importance(
            query, key, scaling=kwargs.get('scaling'), attention_mask=attention_mask))
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

    AttentionInterface.register(ATTENTION_NAME, attention)
    AttentionMaskInterface.register(ATTENTION_NAME, AttentionMaskInterface()['sdpa'])
    previous = decoder.config._attn_implementation
    decoder.config._attn_implementation = ATTENTION_NAME
    for layer in decoder.layers:
        layer.self_attn._rvs_observer = callback

    def cleanup():
        decoder.config._attn_implementation = previous
        for layer in decoder.layers:
            del layer.self_attn._rvs_observer
    return cleanup


class ObservationEngine:
    def __init__(self, model_path, device='cuda:0', dtype='float16'):
        self.model_path = str(Path(model_path).resolve())
        self.device, self.dtype_name = device, dtype
        self.model = self.processor = None

    def load(self, progress):
        if self.model is not None:
            return
        from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; check the rekv environment or use --device cpu --dtype float32")
        progress('loading_model', '加载本地 LLaVA-OneVision 模型（首次计算）', 0.12)
        processor = LlavaOnevisionProcessor.from_pretrained(self.model_path, local_files_only=True)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_path, local_files_only=True, dtype=getattr(torch, self.dtype_name),
            device_map=self.device, attn_implementation='sdpa', low_cpu_mem_usage=True,
        ).eval()
        if model.config.text_config.model_type != 'qwen2':
            raise ValueError("This observer currently supports LLaVA-OneVision Qwen2 checkpoints")
        self.processor, self.model = processor, model

    @torch.inference_mode()
    def observe(self, images, frames, record, params, progress, check_cancel, *, artifact_dir):
        began = time.monotonic()
        self.load(progress)
        check_cancel()
        model, processor = self.model, self.processor
        decoder = model.model.language_model
        embedding = decoder.get_input_embeddings()
        device = embedding.weight.device
        prefix_ids = processor.tokenizer(INIT_PROMPT, return_tensors='pt').input_ids.to(device)
        prefix = embedding(prefix_ids)
        features = []
        processed_shape = None
        for start in range(0, len(images), 4):
            check_cancel()
            progress('vision', f'提取视频特征 {min(start + 4, len(images))}/{len(images)} 帧',
                     0.22 + 0.23 * start / len(images))
            pixels = processor.video_processor(
                videos=images[start:start + 4], do_sample_frames=False, return_tensors='pt',
            ).pixel_values_videos.to(device=model.device, dtype=model.dtype)
            if processed_shape is not None and processed_shape != tuple(pixels.shape[-2:]):
                raise ValueError('Processed frame dimensions changed between vision batches')
            processed_shape = tuple(pixels.shape[-2:])
            output = model.model.get_video_features(
                pixel_values=pixels, vision_feature_layer=model.config.vision_feature_layer,
                vision_feature_select_strategy=model.config.vision_feature_select_strategy,
                return_dict=True,
            )
            features.append(output.pooler_output.to(device))
            del output, pixels
        video_features = torch.cat(features, dim=1)
        if video_features.shape[1] % len(frames):
            raise ValueError("Video features do not divide evenly into frames")
        tokens_per_frame = video_features.shape[1] // len(frames)
        grid = math.isqrt(tokens_per_frame)
        spatial = video_token_layout(processor.video_processor, model.config.vision_config,
                                     processed_shape, tokens_per_frame)
        for frame in frames:
            frame['token_start'] = prefix.shape[1] + frame['index'] * tokens_per_frame
            frame['token_end'] = frame['token_start'] + tokens_per_frame
        parts = [prefix, video_features]
        text_tokens = [dict(position=i, token_id=int(token_id),
                            text=processor.tokenizer.decode([int(token_id)]), kind='prompt')
                       for i, token_id in enumerate(prefix_ids[0])]
        question = record['conversations'][params['question_index']]['question']
        suffix_text = f'\n{question}<|im_end|><|im_start|>assistant\n'
        suffix_ids = processor.tokenizer(suffix_text, return_tensors='pt').input_ids.to(device)
        if params['mode'] == 'question':
            offset = prefix.shape[1] + video_features.shape[1]
            text_tokens.extend(dict(position=offset + i, token_id=int(token_id),
                                    text=processor.tokenizer.decode([int(token_id)]), kind='question')
                               for i, token_id in enumerate(suffix_ids[0]))
            parts.append(embedding(suffix_ids))
        inputs = torch.cat(parts, dim=1)
        length = inputs.shape[1]
        limit = decoder.config.max_position_embeddings
        if length > limit:
            raise ValueError(f"Input has {length} tokens, exceeding model context {limit}; reduce max_frames")
        answer_input_length = length + (suffix_ids.shape[1] if params['mode'] == 'video' else 0)
        if answer_input_length >= limit:
            raise ValueError(f'Question input has {answer_input_length} tokens; reduce max_frames to leave generation space')
        prefix_length = length - WINDOW_SIZE
        scores = [None] * len(decoder.layers)

        def capture(layer, score):
            check_cancel()
            scores[layer] = score
            progress('attention', f'计算注意力：第 {layer + 1}/{len(scores)} 层',
                     0.48 + 0.23 * (layer + 1) / len(scores))

        cleanup = install_observer(decoder, capture)
        try:
            prefill = decoder(inputs_embeds=inputs, use_cache=True, return_dict=True)
        finally:
            cleanup()
        check_cancel()
        if any(score is None for score in scores):
            raise RuntimeError("Some decoder layers did not produce observation scores")
        kv_snapshot = save_kv_snapshot(
            prefill.past_key_values, frames, Path(artifact_dir) / 'kv', check_cancel,
            lambda layer, total: progress('kv_snapshot', f'保存逐帧 K/V：第 {layer}/{total} 层',
                                          0.71 + .06 * layer / total))
        observation_elapsed = time.monotonic() - began
        progress('answer', '基于同一批视频帧和所选问题生成回答', 0.78)
        answer_started = time.monotonic()
        if params['mode'] == 'video':
            prefill = decoder(input_ids=suffix_ids, past_key_values=prefill.past_key_values,
                              use_cache=True, return_dict=True)
        requested_tokens = params['max_new_tokens']
        budget = min(requested_tokens, limit - answer_input_length)
        stop_ids = model.generation_config.eos_token_id
        stop_ids = set(stop_ids if isinstance(stop_ids, (list, tuple)) else [stop_ids])
        stop_ids.add(processor.tokenizer.eos_token_id)
        stop_ids.discard(None)

        def on_token(count):
            if count == 1 or count % 8 == 0:
                progress('answer', f'生成模型回答：{count}/{budget} token', 0.78 + 0.19 * count / budget)

        answer_ids, finish_reason = greedy_token_ids(
            decoder, model.lm_head, prefill, budget, stop_ids, check_cancel, on_token)
        del prefill
        if finish_reason == 'max_new_tokens' and budget < requested_tokens:
            finish_reason = 'context_limit'
        answer = dict(question=question, question_index=params['question_index'],
                      text=processor.tokenizer.decode(answer_ids, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False).strip(),
                      token_ids=answer_ids, generated_tokens=len(answer_ids), max_new_tokens=requested_tokens,
                      input_tokens=answer_input_length, finish_reason=finish_reason, decoding='greedy',
                      elapsed_seconds=time.monotonic() - answer_started,
                      context='same selected video frames + selected question; reference answer is not an input')
        check_cancel()
        progress('statistics', '计算帧内条件熵并保存结果与模型回答', 0.98)
        versions = {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'numpy']}
        result = dict(
            sequence=dict(total_tokens=length, prefix_tokens=prefix_length,
                          prompt_tokens=prefix.shape[1], video_tokens=video_features.shape[1],
                          tokens_per_frame=tokens_per_frame, grid_size=grid if grid * grid == tokens_per_frame else None,
                          window_start=prefix_length, window_end=length, window_size=WINDOW_SIZE,
                          text_tokens=text_tokens),
            model=dict(path=self.model_path, dtype=self.dtype_name, device=self.device,
                       layers=len(scores), heads=decoder.config.num_attention_heads,
                       kv_heads=decoder.config.num_key_value_heads, context_limit=limit, versions=versions),
            method=dict(attention='full causal prefill, native RoPE, no KV eviction',
                        aggregation='mean over the last 32 queries and attention heads; all = mean over layers',
                        importance='softmax over every causally visible key, then retain keys before the window',
                        entropy='p_i = s_i / sum_frame(s); H = -sum(p_i * ln(p_i)); H_norm = H / ln(scored_tokens)',
                        pooling=False, frames_include_window_tokens=True,
                        reference='https://github.com/FasterDecoding/SnapKV'),
            frames=frames, spatial=spatial, answer=answer, kv_snapshot=kv_snapshot,
            views=build_views(np.stack(scores), frames, prefix_length),
            observation_elapsed_seconds=observation_elapsed,
            elapsed_seconds=time.monotonic() - began,
        )
        return result
