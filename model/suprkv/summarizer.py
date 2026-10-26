"""Isolated OneVision captioning; only text crosses into the scoring model."""
from contextlib import nullcontext
from typing import Protocol

import torch
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor

from observation.engine import question_inputs, greedy_token_ids
from observation.segments import SUMMARY_SYSTEM, summary_prompt


DEFAULT_SUMMARIZER = 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf'


class SummarizerLoadError(RuntimeError):
    """Loading is a configuration/startup failure, never a per-segment fallback."""


def validate_summarizer_path(value):
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError('summarizer_model_path must be a nonempty string or null')
    return value.strip() if value is not None else None


class Summarizer(Protocol):
    model_path: str
    context_limit: int
    parameter_bytes: int

    def encode_frame(self, image): ...
    def context_length(self, features, frames): ...
    def generate(self, features, frames, max_new_tokens, check_cancel): ...


class OneVisionSummarizer:
    """Works with an independent model or borrows the main model's native decoder."""
    def __init__(self, model, processor, model_path, decoder_context=nullcontext):
        self.model, self.processor = model, processor
        self.model_path = model_path
        self.decoder = model.model.language_model
        self.tokenizer = processor.tokenizer
        self.device = self.decoder.get_input_embeddings().weight.device
        self.dtype = model.dtype
        self.context_limit = self.decoder.config.max_position_embeddings
        self.parameter_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
        self.decoder_context = decoder_context

    @classmethod
    def load(cls, model_path, device, dtype):
        try:
            processor = LlavaOnevisionProcessor.from_pretrained(model_path)
            model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_path, dtype=dtype, device_map=str(device),
                attn_implementation='sdpa', low_cpu_mem_usage=True).eval()
        except torch.cuda.OutOfMemoryError:
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            raise SummarizerLoadError(f'Cannot load summarizer {model_path}: {exc}') from exc
        return cls(model, processor, model_path)

    @torch.inference_mode()
    def encode_frame(self, image):
        pixels = self.processor.video_processor(videos=[image], do_sample_frames=False,
            return_tensors='pt').pixel_values_videos.to(self.device, self.dtype)
        output = self.model.model.get_video_features(pixel_values=pixels,
            vision_feature_layer=self.model.config.vision_feature_layer,
            vision_feature_select_strategy=self.model.config.vision_feature_select_strategy,
            return_dict=True)
        return output.pooler_output.reshape(1, -1, self.decoder.config.hidden_size).detach().clone()

    def prompt_ids(self, frames):
        system = f'<|im_start|>system\n{SUMMARY_SYSTEM}<|im_end|><|im_start|>user\n'
        ids = self.tokenizer(system, return_tensors='pt').input_ids.to(self.device)
        suffix = question_inputs(self.tokenizer, summary_prompt(frames)).to(self.device)
        return ids, suffix

    def context_length(self, features, frames):
        ids, suffix = self.prompt_ids(frames)
        return ids.shape[1] + suffix.shape[1] + sum(f.shape[1] for f in features)

    def inputs(self, features, frames):
        ids, suffix = self.prompt_ids(frames)
        embedding = self.decoder.get_input_embeddings()
        visual = torch.cat([f.to(self.device, self.dtype) for f in features], 1)
        return torch.cat([embedding(ids), visual, embedding(suffix)], 1)

    @torch.inference_mode()
    def generate(self, features, frames, max_new_tokens, check_cancel):
        check_cancel()
        with self.decoder_context():
            inputs = self.inputs(features, frames)
            output = self.decoder(inputs_embeds=inputs, use_cache=True, return_dict=True)
            ids, _ = greedy_token_ids(self.decoder, self.model.lm_head, output,
                max_new_tokens, {self.tokenizer.eos_token_id}, check_cancel)
        return self.tokenizer.decode(ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False).strip()
