"""Uncompressed causal prefill of local LLaVA-OneVision with a window observer."""
import importlib.metadata
from contextlib import nullcontext
import math
from pathlib import Path
import time

import numpy as np
import torch

from observation.metrics import (WINDOW_SIZE, build_views,
                                 window_importance, validate_video_window)
from observation.spatial import video_token_layout
from observation.segments import (EPSILON, SUMMARY_SYSTEM, TokenStateStore, capture_token_states,
                                  selected_vision_features, vision_frame_distances, segment_by_ema, summary_prompt)


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


def question_inputs(tokenizer, question):
    """Tokenize the question suffix used for model answering."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question body must contain at least one non-whitespace token")
    suffix = f'\n{question}<|im_end|><|im_start|>assistant\n'
    return tokenizer(suffix, return_tensors='pt').input_ids


def install_observer(decoder, callback, *, window=WINDOW_SIZE,
                     check_cancel=lambda: None):
    """Dispatch post-RoPE Q/K to the observer, preserving native SDPA outputs."""
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import AttentionMaskInterface

    def attention(module, query, key, value, attention_mask, **kwargs):
        if kwargs.get('sliding_window') is not None:
            raise ValueError("This experiment requires full causal attention")
        check_cancel()
        scores = window_importance(query, key, scaling=kwargs.get('scaling'),
                                   attention_mask=attention_mask, window=window,
                                   check_cancel=check_cancel)
        module._rvs_observer(module.layer_idx, scores)
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
    def observe(self, images, frames, record, params, progress, check_cancel, artifact_dir=None):
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
        segment_enabled = params.get('segment_enabled', True)
        vision_distances, previous_vision = [], None
        vision_shape = None
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
            if segment_enabled:
                raw_vision = selected_vision_features(output, model.config.vision_feature_layer,
                                                     model.config.vision_feature_select_strategy)
                distances, previous_vision = vision_frame_distances(raw_vision, previous_vision)
                vision_distances.extend(distances.tolist())
                vision_shape = tuple(raw_vision.shape[1:])
                del raw_vision
            del output, pixels
        del previous_vision
        video_features = torch.cat(features, dim=1)
        video_window = validate_video_window(params.get('video_window_size', WINDOW_SIZE),
                                             video_features.shape[1])
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
        suffix_ids = question_inputs(processor.tokenizer, question)
        suffix_ids = suffix_ids.to(device)
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
        window = video_window if params['mode'] == 'video' else WINDOW_SIZE
        prefix_length = length - window
        scores = [None] * len(decoder.layers)
        if segment_enabled and artifact_dir is None:
            raise ValueError('Segment token observations require an artifact directory')
        token_stores = {kind: TokenStateStore(artifact_dir, kind) for kind in ('key', 'hidden')} if segment_enabled else {}

        def capture_video_states(kind, layer, states):
            token_stores[kind].video(layer, states, frames)

        def capture(layer, captured):
            check_cancel()
            scores[layer] = captured
            progress('attention', f'计算注意力：第 {layer + 1}/{len(scores)} 层',
                     0.48 + 0.30 * (layer + 1) / len(scores))

        cleanup = install_observer(decoder, capture, window=window, check_cancel=check_cancel)
        try:
            with capture_token_states(decoder, capture_video_states, check_cancel) if segment_enabled else nullcontext():
                prefill = decoder(inputs_embeds=inputs, use_cache=True, return_dict=True)
        finally:
            cleanup()
        check_cancel()
        if any(score is None for score in scores):
            raise RuntimeError("Some decoder layers did not produce observation scores")
        if params['mode'] == 'video':
            prefill = decoder(input_ids=suffix_ids, past_key_values=prefill.past_key_values,
                              use_cache=True, return_dict=True)
        observation_elapsed = time.monotonic() - began
        progress('answer', '基于同一批视频帧和所选问题生成回答', 0.82)
        answer_started = time.monotonic()
        requested_tokens = params['max_new_tokens']
        budget = min(requested_tokens, limit - answer_input_length)
        stop_ids = model.generation_config.eos_token_id
        stop_ids = set(stop_ids if isinstance(stop_ids, (list, tuple)) else [stop_ids])
        stop_ids.add(processor.tokenizer.eos_token_id)
        stop_ids.discard(None)

        def on_token(count):
            if count == 1 or count % 8 == 0:
                progress('answer', f'生成模型回答：{count}/{budget} token', 0.82 + 0.05 * count / budget)

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
        segment_experiment = None
        if segment_enabled:
            segment_experiment = segment_by_ema(
                vision_distances, frames, params.get('segment_ema_alpha', .2),
                params.get('segment_threshold_multiplier', 1.5), params.get('segment_min_frames', 1))
            segment_experiment.update(
                segmentation_space='vision encoder selected hidden states, before multimodal projector and spatial pooling',
                vision_feature_layer=model.config.vision_feature_layer,
                vision_feature_select_strategy=model.config.vision_feature_select_strategy,
                vision_tokens_per_frame=vision_shape[0], vision_feature_dim=vision_shape[1], epsilon=EPSILON,
                distance='mean over aligned vision patches of L2(V[t]-V[t-1])/sqrt(mean_i ||V[t,i]||^2 + epsilon)',
                threshold_rule='D[t] > max(epsilon, multiplier * EMA[t-1]); update EMA after decision, including boundaries',
                warmup='First adjacent pair initializes EMA and cannot create a boundary; final segment may be shorter than min_frames',
                pooling=False,
                similarity='Per-token cosine in selected key or hidden space; all = per-cell mean over language-model layers',
                token_similarity=dict(storage='per-layer unpooled states in token_keys/*.npy and token_hidden/*.npy',
                                      representations=['key', 'hidden'],
                                      definitions=dict(key='pre-RoPE k_proj output; concatenate native KV heads',
                                                       hidden='decoder block output after attention and MLP residuals, before final model norm'),
                                      rows='summary content tokens', columns='selected frame spatial tokens in row-major order'),
                interpretation='Representation-space observation, not a calibrated semantic score; visual language-model states retain earlier-frame context',
                system_prompt=SUMMARY_SYSTEM)
            self.summarize_segments(segment_experiment, token_stores, video_features,
                                    tokens_per_frame, frames, params, stop_ids, progress, check_cancel)
        progress('statistics', '汇总 attention mass 并保存结果与模型回答', 0.98)
        versions = {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'numpy']}
        result = dict(
            sequence=dict(total_tokens=length, prefix_tokens=prefix_length,
                          prompt_tokens=prefix.shape[1], video_tokens=video_features.shape[1],
                          tokens_per_frame=tokens_per_frame, grid_size=grid if grid * grid == tokens_per_frame else None,
                          window_start=prefix_length, window_end=length, window_size=window,
                          text_tokens=text_tokens),
            model=dict(path=self.model_path, dtype=self.dtype_name, device=self.device,
                       layers=len(scores), heads=decoder.config.num_attention_heads,
                       kv_heads=decoder.config.num_key_value_heads, context_limit=limit, versions=versions),
            method=dict(attention='full causal prefill, native RoPE, no KV eviction',
                        aggregation=f'mean over the last {window} queries and attention heads; all = mean over layers',
                        importance='softmax over every causally visible key, then retain keys before the window',
                        pooling=False, frames_include_window_tokens=True,
                        reference='https://github.com/FasterDecoding/SnapKV'),
            frames=frames, spatial=spatial, answer=answer,
            views=build_views(np.stack(scores), frames, prefix_length),
            observation_elapsed_seconds=observation_elapsed,
            elapsed_seconds=time.monotonic() - began,
        )
        if segment_experiment is not None:
            result['segment_experiment'] = segment_experiment
        return result

    @torch.inference_mode()
    def summarize_segments(self, experiment, token_stores, video_features, tokens_per_frame,
                           frames, params, stop_ids, progress, check_cancel):
        """Generate each summary from its segment only, then encode text independently."""
        model, tokenizer = self.model, self.processor.tokenizer
        decoder = model.model.language_model
        embedding = decoder.get_input_embeddings()
        device, limit = embedding.weight.device, decoder.config.max_position_embeddings
        system = f'<|im_start|>system\n{SUMMARY_SYSTEM}<|im_end|><|im_start|>user\n'
        system_ids = tokenizer(system, return_tensors='pt').input_ids.to(device)
        text_prefix = tokenizer(INIT_PROMPT, return_tensors='pt').input_ids.to(device)
        requested = params.get('summary_max_new_tokens', 192)
        for index, segment in enumerate(experiment['segments']):
            check_cancel()
            began = time.monotonic()
            start, end = segment['start_frame'], segment['end_frame']
            prompt = summary_prompt(frames[start:end])
            suffix = question_inputs(tokenizer, prompt).to(device)
            visual = video_features[:, start * tokens_per_frame:end * tokens_per_frame]
            inputs = torch.cat([embedding(system_ids), visual, embedding(suffix)], dim=1)
            input_length = inputs.shape[1]
            if input_length >= limit:
                raise ValueError(f'Segment {index + 1} summary prompt exceeds context; reduce max_frames')
            budget = min(requested, limit - input_length)

            def report(count=0):
                progress('segment_summary',
                         f'生成 segment {index + 1}/{len(experiment["segments"])} 总结：{count}/{budget} token',
                         .88 + .09 * (index + count / budget) / len(experiment['segments']))

            report()
            output = decoder(inputs_embeds=inputs, use_cache=True, return_dict=True)
            ids, reason = greedy_token_ids(decoder, model.lm_head, output, budget, stop_ids,
                                           check_cancel, lambda n: report(n) if n == 1 or n % 8 == 0 else None)
            del output, inputs
            if reason == 'max_new_tokens' and budget < requested:
                reason = 'context_limit'
            text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            segment['summary'] = dict(text=text, prompt=prompt, token_ids=ids, generated_tokens=len(ids),
                                      input_tokens=input_length, max_new_tokens=requested, finish_reason=reason,
                                      decoding='greedy', context='this segment only; no question, answer or other summaries',
                                      elapsed_seconds=time.monotonic() - began, tokens=[])
            if text:
                # Re-tokenize only the visible summary, excluding EOS and chat markers.
                content = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
                combined = torch.cat([text_prefix, content], dim=1)
                segment['summary']['tokens'] = [dict(index=i, token_id=int(token_id),
                    text=tokenizer.decode([int(token_id)], skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False))
                    for i, token_id in enumerate(content[0])]
                if combined.shape[1] > limit:
                    raise ValueError('Summary text exceeds context for independent encoding')

                def capture_text(kind, layer, states):
                    token_stores[kind].summary(layer, index, states[0, text_prefix.shape[1]:])

                if content.shape[1]:
                    with capture_token_states(decoder, capture_text, check_cancel):
                        encoded = decoder(input_ids=combined, use_cache=False, return_dict=True)
                    del encoded
