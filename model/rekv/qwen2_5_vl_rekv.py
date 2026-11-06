"""Qwen2.5-VL backbone with ReKV's frame retrieval and 1D cache positions."""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from .qwen2_5_vl_vision import FRAME_TOKENS, frame_features
from model.video_backbone import INITIAL_PROMPTS, question_prompt
from model.loading import load_pretrained
from .abstract_rekv import Abstract_ReKV
from .llava_onevision_rekv import LlavaOneVision_ReKV
from .patch import patch_hf


class Qwen2_5_VL_ReKV(Qwen2_5_VLForConditionalGeneration, Abstract_ReKV):
    def __init__(self, config, processor, n_frame_tokens, init_prompt_ids,
                 n_local, topk, chunk_size):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids,
                              n_local, topk, chunk_size)

    @property
    def language_model(self):
        return self.model.language_model

    def get_prompt(self, query, mc=False):
        return question_prompt('qwen2_5_vl', query, mc)

    def _encode_video_chunk(self, video_chunk):
        features = frame_features(self, self.processor, video_chunk).flatten(0, 1)[None]
        if features.shape[1] > self.n_local:
            raise ValueError('Video chunk exceeds n_local')
        output = self.language_model(inputs_embeds=features,
            past_key_values=self.kv_cache, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values

    @torch.inference_mode()
    def encode_video(self, video, encode_chunk_size=64):
        if encode_chunk_size < 1:
            raise ValueError('encode_chunk_size must be positive')
        # Respect the cache window even if a caller requests a larger chunk.
        size = min(encode_chunk_size, self.n_local // self.n_frame_tokens)
        if size < 1:
            raise ValueError('n_local must fit at least one frame')
        return Abstract_ReKV.encode_video(self, video, size)

    # Both models expose the same bare decoder/lm_head and Qwen chat tokenizer.
    question_answering = LlavaOneVision_ReKV.question_answering


def load_model(model_path='Qwen/Qwen2.5-VL-7B-Instruct', n_init=None,
               n_local=15000, topk=64, chunk_size=1):
    n_local = 15000 if n_local is None else n_local
    if n_local < FRAME_TOKENS:
        raise ValueError(f'Qwen2.5-VL requires n_local >= {FRAME_TOKENS}')
    if not 1 <= chunk_size <= topk:
        raise ValueError('Require 1 <= chunk_size <= topk')
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    ids = processor.tokenizer(INITIAL_PROMPTS['qwen2_5_vl'], return_tensors='pt').input_ids
    model = load_pretrained(
        Qwen2_5_VL_ReKV, model_path, dtype=torch.float16,
        processor=processor, n_frame_tokens=FRAME_TOKENS, init_prompt_ids=ids,
        n_local=n_local, topk=topk, chunk_size=chunk_size)
    model.init_prompt_ids = ids.to(model.get_input_embeddings().weight.device)
    patch_hf(model.language_model, n_init=ids.shape[1] if n_init is None else n_init,
             n_local=n_local, topk=topk, chunk_size=chunk_size,
             block_size=FRAME_TOKENS, max_cached_block=128,
             exc_block_size=FRAME_TOKENS, fattn=True, pin_memory=True)
    return model.eval(), processor
