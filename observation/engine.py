"""Uncompressed causal prefill of local LLaVA-OneVision with a window observer."""
import importlib.metadata
import math
from pathlib import Path
import time

import numpy as np
import torch

from observation.metrics import (WINDOW_SIZE, build_views, build_comparison, query_importance,
                                 window_importance, validate_video_window)
from observation.spatial import video_token_layout
from observation.kv import PreRoPEKeyCapture, save_kv_snapshot
from observation.vendi import score_snapshot


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


def question_inputs(tokenizer, question, offset):
    """Tokenize the exact suffix once; select tokens overlapping the question body."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question body must contain at least one non-whitespace token")
    suffix = f'\n{question}<|im_end|><|im_start|>assistant\n'
    encoded = tokenizer(suffix, return_tensors='pt', return_offsets_mapping=True)
    spans = encoded.offset_mapping[0].tolist()
    selected = [i for i, (start, end) in enumerate(spans)
                if end > 1 and start < 1 + len(question) and end > start]
    if not selected:
        raise ValueError("Question body did not produce any query tokens")
    ids = encoded.input_ids[0].tolist()
    tokens = [dict(position=offset + i, token_id=ids[i], text=tokenizer.decode([ids[i]]),
                   char_start=spans[i][0] - 1, char_end=spans[i][1] - 1) for i in selected]
    return encoded.input_ids, dict(kind='question_body', positions=[offset + i for i in selected],
                                   count=len(selected), tokens=tokens, text=question)


def install_observer(decoder, callback, *, window=WINDOW_SIZE, selections=None,
                     check_cancel=lambda: None):
    """Dispatch post-RoPE Q/K to the observer, preserving native SDPA outputs."""
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import AttentionMaskInterface

    def attention(module, query, key, value, attention_mask, **kwargs):
        if kwargs.get('sliding_window') is not None:
            raise ValueError("This experiment requires full causal attention")
        check_cancel()
        if selections is None:
            scores = window_importance(query, key, scaling=kwargs.get('scaling'),
                                       attention_mask=attention_mask, window=window)
        else:
            scores = {}
            # Reuse coincident selections (legacy video window == comparison before).
            computed = {}
            for name, positions in selections.items():
                indices = tuple(positions)
                if indices not in computed:
                    computed[indices] = query_importance(
                        query, key, indices, scaling=kwargs.get('scaling'),
                        attention_mask=attention_mask, check_cancel=check_cancel)
                scores[name] = computed[indices]
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
        video_window = validate_video_window(params.get('video_window_size', WINDOW_SIZE),
                                             video_features.shape[1])
        video_end = prefix.shape[1] + video_features.shape[1]
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
        suffix_ids, question_queries = question_inputs(processor.tokenizer, question, video_end)
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
        before_scores = [None] * len(decoder.layers)
        after_scores = [None] * len(decoder.layers)
        before_positions = list(range(video_end - video_window, video_end))
        queries = dict(
            before=dict(kind='video_tail', positions=before_positions, count=video_window,
                        tokens=[dict(position=i, frame_index=(i - prefix.shape[1]) // tokens_per_frame,
                                     frame_token=(i - prefix.shape[1]) % tokens_per_frame)
                                for i in before_positions]),
            after=question_queries,
        )

        def capture(layer, captured):
            check_cancel()
            scores[layer] = captured['legacy']
            before_scores[layer] = captured['before'][:video_end]
            if 'after' in captured:
                after_scores[layer] = captured['after']
            progress('attention', f'计算注意力：第 {layer + 1}/{len(scores)} 层',
                     0.48 + 0.23 * (layer + 1) / len(scores))

        selections = dict(legacy=range(prefix_length, length), before=before_positions)
        if params['mode'] == 'question':
            selections['after'] = question_queries['positions']
        pre_rope_capture = PreRoPEKeyCapture(
            decoder, frames, Path(artifact_dir) / 'key_pre_rope', check_cancel,
            lambda layer, total: progress('pre_rope_key', f'保存 RoPE 前 Key：第 {layer}/{total} 层',
                                          0.48 + .23 * (layer - 1) / total))
        cleanup = install_observer(decoder, capture, selections=selections, check_cancel=check_cancel)
        try:
            with pre_rope_capture:
                prefill = decoder(inputs_embeds=inputs, use_cache=True, return_dict=True)
        finally:
            cleanup()
        pre_rope_key_snapshot = pre_rope_capture.manifest()
        check_cancel()
        if any(score is None for score in scores):
            raise RuntimeError("Some decoder layers did not produce observation scores")
        kv_snapshot = save_kv_snapshot(
            prefill.past_key_values, frames, Path(artifact_dir) / 'kv', check_cancel,
            lambda layer, total: progress('kv_snapshot', f'保存逐帧 K/V：第 {layer}/{total} 层',
                                          0.71 + .06 * layer / total))
        if params['mode'] == 'video':
            def capture_question(layer, captured):
                check_cancel()
                after_scores[layer] = captured['after']
                progress('question_attention', f'问题正文注意力：第 {layer + 1}/{len(scores)} 层',
                         0.77 + .05 * (layer + 1) / len(scores))

            cleanup = install_observer(
                decoder, capture_question, selections=dict(after=question_queries['positions']),
                check_cancel=check_cancel)
            try:
                prefill = decoder(input_ids=suffix_ids, past_key_values=prefill.past_key_values,
                                  use_cache=True, return_dict=True)
            finally:
                cleanup()
        if any(score is None for score in before_scores + after_scores):
            raise RuntimeError("Some decoder layers did not produce comparison scores")
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
                progress('answer', f'生成模型回答：{count}/{budget} token', 0.82 + 0.15 * count / budget)

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
        progress('uniqueness', '计算逐帧 K/V 最近邻冗余比例 R_NN', 0.97)
        uniqueness = score_snapshot(
            Path(artifact_dir) / 'kv', kv_snapshot, check_cancel,
            lambda layer, total: progress('uniqueness', f'计算 RoPE 后 uniqueness：第 {layer}/{total} 层',
                                          0.97 + .005 * layer / total))
        uniqueness_pre_rope = score_snapshot(
            Path(artifact_dir) / 'key_pre_rope', pre_rope_key_snapshot, check_cancel,
            lambda layer, total: progress('uniqueness', f'计算 RoPE 前 uniqueness：第 {layer}/{total} 层',
                                          0.975 + .005 * layer / total))
        progress('statistics', '计算帧内条件熵并保存结果与模型回答', 0.98)
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
                        entropy='p_i = s_i / sum_frame(s); H = -sum(p_i * ln(p_i)); H_norm = H / ln(scored_tokens)',
                        pooling=False, frames_include_window_tokens=True,
                        reference='https://github.com/FasterDecoding/SnapKV'),
            frames=frames, spatial=spatial, answer=answer, kv_snapshot=kv_snapshot,
            uniqueness=uniqueness, uniqueness_pre_rope=uniqueness_pre_rope,
            pre_rope_key_snapshot=pre_rope_key_snapshot,
            views=build_views(np.stack(scores), frames, prefix_length),
            attention_comparison=build_comparison(before_scores, after_scores, frames, queries),
            observation_elapsed_seconds=observation_elapsed,
            elapsed_seconds=time.monotonic() - began,
        )
        return result
