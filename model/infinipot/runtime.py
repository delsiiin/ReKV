"""Qwen2.5-VL runtime retaining the released InfiniPot-V execution flow."""
from dataclasses import asdict
from types import MethodType

import torch

from model.loading import load_pretrained
from .config import InfiniPotConfig
from .core import InfiniPotCore
from .rope import source_rope_index


def _reference_rope(self, input_ids, mm_token_type_ids=None, **kwargs):
    # Transformers 5 passes modality IDs; the 4.57.3 implementation infers them.
    allowed = ('image_grid_thw', 'video_grid_thw', 'second_per_grid_ts', 'attention_mask')
    return source_rope_index(self, input_ids, **{k: v for k, v in kwargs.items() if k in allowed})


class InfiniPotRuntime(InfiniPotCore):
    def __init__(self, model, processor, config=None, *, max_new_tokens=None):
        self.config = config or InfiniPotConfig()
        self.__dict__.update(asdict(self.config))
        self.model, self.processor = model, processor
        self.MAX_GEN_TOKENS = 300 if max_new_tokens is None else max_new_tokens
        if self.MAX_GEN_TOKENS < 1:
            raise ValueError('max_new_tokens must be positive')
        # This model belongs solely to this runtime. Restore on close for callers
        # that inject a model instance; no global class/attention patch is made.
        self._original_rope = model.model.get_rope_index
        model.model.get_rope_index = MethodType(_reference_rope, model.model)
        self.reset()

    @classmethod
    def load(cls, model_path, config=None, *, max_new_tokens=None):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        model = load_pretrained(Qwen2_5_VLForConditionalGeneration,
                               model_path, dtype='auto', attn_implementation='flash_attention_2')
        return cls(model.eval(), AutoProcessor.from_pretrained(model_path), config,
                   max_new_tokens=max_new_tokens)

    def reset(self):
        self.video = None
        self.model.model.rope_deltas = None

    def close(self):
        self.reset()
        self.model.model.get_rope_index = self._original_rope

    def set_video(self, video):
        self.video = video

    def get_prompt(self, question, mc=False):
        # The source uses apply_chat_template, without ReKV's answer prefill.
        return question

    def prepare_inputs(self, question):
        content = ([{'type': 'video', 'video': 'visible-prefix'}] if self.video is not None else [])
        content.append({'type': 'text', 'text': question})
        text = self.processor.apply_chat_template(
            [{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
        if self.video is None:
            return self.processor(text=[text], return_tensors='pt').to(self.model.device)
        inputs = self.processor(text=[text], images=None, videos=[self.video], padding=True,
                                do_sample_frames=False, return_tensors='pt').to(self.model.device)
        # As in prepare_video_input: reconstruct the full contiguous placeholder
        # span from tokenizer IDs and the patch merger's factor of four.
        ids = self.processor.tokenizer(text, return_tensors='pt').input_ids.to(self.model.device)
        start = (ids[0] == self.model.config.vision_start_token_id).nonzero().flatten()
        end = (ids[0] == self.model.config.vision_end_token_id).nonzero().flatten()
        if start.numel() != 1 or end.numel() != 1:
            raise ValueError('InfiniPot expects exactly one video in the chat template')
        video_length = inputs['pixel_values_videos'].shape[0] // 4
        pads = ids.new_full((1, video_length), self.model.config.video_token_id)
        inputs['input_ids'] = torch.cat([ids[:, :int(start[0]) + 1], pads, ids[:, int(end[0]):]], dim=-1)
        inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape, device=self.model.device)
        # New Transformers requires this field, reconstructed after replacement.
        inputs['mm_token_type_ids'] = (inputs['input_ids'] == self.model.config.video_token_id).long() * 2
        return inputs

    @torch.inference_mode()
    def generate(self, inputs):
        # Fix the source's undefined MAX_GEN_TOKENS global to the class constant.
        inputs.pop('second_per_grid_ts', None)
        generated = self.model.generate(**inputs, max_new_tokens=self.MAX_GEN_TOKENS)
        return self.processor.batch_decode(
            [out[len(ids):] for ids, out in zip(inputs['input_ids'], generated)],
            skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=None):
        # BaseVQA's 16/256/1024 limits must not override the reference's 300.
        self.model.model.rope_deltas = None
        try:
            inputs = self.prepare_inputs(input_text['prompt'])
            frames = int(inputs['video_grid_thw'][0, 0]) if 'video_grid_thw' in inputs else 0
            if self.block_size > 0 and frames > self.block_size and self.compress_frame_num > 0:
                tokens = int((inputs['input_ids'] == self.model.config.video_token_id).sum()) // frames
                self.config.validate_budget(tokens, total_frames=frames)
                return self.block_process(inputs)
            return self.generate(inputs)
        finally:
            self.model.model.rope_deltas = None
