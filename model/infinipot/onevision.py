"""InfiniPot's block schedule and TaR/VaN selection for LLaVA-OneVision.

Targets the repository's Transformers 5.5.4 API. Each sampled frame contributes
one pooled spatial grid; the native video newline is appended once, after all
frames. Cached keys keep their original one-dimensional RoPE positions.
"""
import copy
import math
from types import SimpleNamespace

import torch
from transformers import DynamicCache

from model.loading import load_pretrained
from model.video_backbone import preprocess_video, video_features
from .cache import process_kv_cache
from .config import InfiniPotConfig
from .reuse import cached_generate_inputs, clear_prefill_cache


class InfiniPotOneVisionRuntime:
    video_loading_options = dict(frame_factor=1, resize=False)

    def __init__(self, model, processor, config=None, *, max_new_tokens=None):
        self.model, self.processor = model, processor
        self.config = config or InfiniPotConfig()
        self.MAX_GEN_TOKENS = 300 if max_new_tokens is None else max_new_tokens
        if self.MAX_GEN_TOKENS < 1:
            raise ValueError('max_new_tokens must be positive')
        self.reset()

    @classmethod
    def load(cls, model_path, config=None, *, max_new_tokens=None):
        from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor
        model = load_pretrained(LlavaOnevisionForConditionalGeneration, model_path,
                               dtype='auto', attn_implementation='flash_attention_2')
        return cls(model.eval(), LlavaOnevisionProcessor.from_pretrained(model_path),
                   config, max_new_tokens=max_new_tokens)

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    def reset(self):
        self.video = None
        self._clear_prefill_cache()

    def _clear_prefill_cache(self):
        clear_prefill_cache(self)

    def close(self):
        self.reset()

    def set_video(self, video):
        # A new visible prefix can change sampling and the tail compression
        # budget. Even re-submitting the same tensor explicitly invalidates it.
        self._clear_prefill_cache()
        self.video = video

    def get_prompt(self, question, mc=False):
        return question

    def _prompt(self, question):
        content = [{'type': 'video'}] if self.video is not None else []
        content.append({'type': 'text', 'text': question})
        return self.processor.apply_chat_template(
            [{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)

    def prepare_inputs(self, question):
        text = self._prompt(question)
        kwargs = {} if self.video is None else dict(videos=[self.video], do_sample_frames=False)
        return self.processor(text=[text], return_tensors='pt', **kwargs).to(
            device=self.device, dtype=self.model.dtype)

    @torch.inference_mode()
    def generate(self, inputs):
        prepared = cached_generate_inputs(self, inputs, self.model.config.video_token_index)
        generated = self.model.generate(**prepared, max_new_tokens=self.MAX_GEN_TOKENS)
        return self.processor.batch_decode(
            [out[len(ids):] for ids, out in zip(inputs['input_ids'], generated)],
            skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def _encode_frames(self, start, end):
        # Process only this block on the GPU; avoid constructing all video
        # embeddings or a full uncompressed KV cache before selection.
        pixels = preprocess_video(self.processor, 'onevision', self.video[start:end])
        pixels = pixels.to(device=self.device, dtype=self.model.dtype)
        return video_features(self.model, 'onevision', pixels).to(self.device)

    def _forward(self, embeddings, cache, position):
        positions = torch.arange(position, position + embeddings.shape[1],
                                 device=self.device).unsqueeze(0)
        return self.model.model.language_model(
            inputs_embeds=embeddings, past_key_values=cache, position_ids=positions,
            use_cache=True, return_dict=True)

    def _prefill_video(self, prefix_ids):
        c = self.config
        total_frames = len(self.video)
        compress = c.block_size > 0 and total_frames > c.block_size and c.compress_frame_num > 0
        prefix = self.model.get_input_embeddings()(prefix_ids)
        system_size = prefix.shape[1]
        cache = DynamicCache()
        start, token_per_frame = 0, None
        while start < total_frames:
            count = total_frames
            if compress:
                count = c.block_size if start == 0 else c.block_size - c.compress_frame_num
            end = min(start + count, total_frames)
            features = self._encode_frames(start, end)
            current_tokens, remainder = divmod(features.shape[1], end - start)
            if remainder or (token_per_frame is not None and current_tokens != token_per_frame):
                raise ValueError('OneVision must produce a fixed number of tokens per frame')
            token_per_frame = current_tokens
            if start == 0:
                if compress:
                    c.validate_budget(token_per_frame, total_frames=total_frames)
                side = math.isqrt(token_per_frame)
                if side * side != token_per_frame:
                    raise ValueError('OneVision pooled video features must form a square spatial grid')
                # The shared selector needs the spatial size for optional pooling.
                # Keep this metadata local instead of changing the model config.
                compression_model = SimpleNamespace(
                    model=self.model.model, config=SimpleNamespace(height_width=side))
                embeddings = torch.cat((prefix, features), dim=1)
                position = 0
            else:
                embeddings = features
                position = system_size + start * token_per_frame
            output = self._forward(embeddings, cache, position)
            cache = output.past_key_values
            if compress and end < total_frames:
                keep = max(c.compress_frame_num, c.block_size - (total_frames - end))
                cache, _ = process_kv_cache(
                    cache, compression_model, system_size, 0, token_per_frame, keep,
                    method=c.compression_method, tar_ratio=c.tar_ratio, query_ratio=c.query_ratio,
                    adaptive_pooling=c.adaptive_pooling, per_frame=c.per_frame,
                    is_first_block=start == 0, is_last_block=False)
            start = end
        return cache, system_size + total_frames * token_per_frame

    def _get_video_prefill(self, prefix_ids):
        # Only the text before the video affects its causal prefill. Questions
        # after the video may differ without repeating vision/LLM computation.
        if (self._prefill_mode != 'block' or self._prefill_config != self.config
                or not torch.equal(self._prefill_prefix_ids, prefix_ids)):
            self._clear_prefill_cache()
            cache, position = self._prefill_video(prefix_ids)
            self._prefill_prefix_ids = prefix_ids.clone()
            self._prefill_position = position
            self._prefill_config = self.config
            self._prefill_cache = cache
            self._prefill_mode = 'block'
        # Generation mutates DynamicCache. Clone tensors AND cache metadata so
        # questions (including a failed generation) cannot alter the snapshot.
        return copy.deepcopy(self._prefill_cache), self._prefill_position

    @torch.inference_mode()
    def block_process(self, input_ids):
        markers = (input_ids[0] == self.model.config.video_token_index).nonzero().flatten()
        if markers.numel() != 1:
            raise ValueError('InfiniPot expects exactly one video in the chat template')
        split = int(markers[0])
        cache, position = self._get_video_prefill(input_ids[:, :split])
        suffix = self.model.get_input_embeddings()(input_ids[:, split + 1:])
        # HF OneVision appends one learned newline per video, not per block.
        newline = self.model.model.image_newline.reshape(1, 1, -1).to(suffix)
        embeddings = torch.cat((newline, suffix), dim=1)
        stop = self.model.generation_config.eos_token_id
        if stop is None:
            stop = self.processor.tokenizer.eos_token_id
        stop = set(stop if isinstance(stop, (list, tuple)) else [stop])
        tokens = []
        for _ in range(self.MAX_GEN_TOKENS):
            output = self._forward(embeddings, cache, position)
            cache = output.past_key_values
            position += embeddings.shape[1]
            logits = self.model.lm_head(output.last_hidden_state[:, -1:])
            token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens.append(int(token[0, 0]))
            if tokens[-1] in stop:
                break
            embeddings = self.model.get_input_embeddings()(token)
        return self.processor.tokenizer.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=None):
        # Match the Qwen runtime: only the explicit CLI override changes 300.
        frames = 0 if self.video is None else len(self.video)
        c = self.config
        if c.block_size > 0 and frames > c.block_size and c.compress_frame_num > 0:
            text = self._prompt(input_text['prompt'])
            ids = self.processor.tokenizer(text, return_tensors='pt').input_ids.to(self.device)
            return self.block_process(ids)
        return self.generate(self.prepare_inputs(input_text['prompt']))
