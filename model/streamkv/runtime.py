"""Source StreamKV algorithm with a Transformers 5 OneVision API bridge.

Segmentation, compression, retrieval, generation and CUDA cache management are
the supplied reference implementations. This module only bridges model APIs.
"""
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from model.video_backbone import question_prompt
from .abstract_streamkv import Abstract_StreamKV
from .config import StreamKVConfig
from .generation import StreamKVGeneration
from .prompts import GUIDANCE_PROMPT, INITIAL_PROMPT


class _ProcessorBridge:
    """Keep the reference's batched/framewise preprocessing call boundaries."""
    def __init__(self, processor):
        self.tokenizer = processor.tokenizer
        self._processor = processor

    def video_processor(self, video, **kwargs):
        return self._processor.video_processor(videos=video, do_sample_frames=False, **kwargs)


class _LanguageModelBridge:
    """Reference CausalLM contract over the Transformers 5 bare text decoder."""
    def __init__(self, model):
        self.model = model
        self.decoder = model.model.language_model

    def __call__(self, input_ids=None, inputs_embeds=None, **kwargs):
        if input_ids is not None:
            text = self.decoder.embed_tokens(input_ids)
            if hasattr(self.decoder.config, 'scale_emb'):
                text = text * self.decoder.config.scale_emb
            # The reference patch explicitly allows video embeds + guidance IDs.
            inputs_embeds = text if inputs_embeds is None else torch.cat((inputs_embeds, text), dim=-2)
        kwargs.pop('return_dict', None)
        output = self.decoder(inputs_embeds=inputs_embeds, return_dict=True, **kwargs)
        logits = self.model.lm_head(output.last_hidden_state).float()
        return CausalLMOutputWithPast(logits=logits, past_key_values=output.past_key_values,
                                     hidden_states=output.hidden_states, attentions=output.attentions)


class StreamKVRuntime(StreamKVGeneration, Abstract_StreamKV):
    def __init__(self, model, processor, config=None, *, n_frame_tokens=196,
                 n_init=None, n_encode=None, fattn=True, async_global_stream=True,
                 pin_memory=True):
        from model.rekv.patch import patch_hf
        from .attention import streamkv_attention_forward

        self.model = model.eval()
        self.config = config or StreamKVConfig()
        self.decoder = model.model.language_model
        self.language_model = _LanguageModelBridge(model)
        self.tokenizer = processor.tokenizer
        init_ids = self.tokenizer(INITIAL_PROMPT, return_tensors='pt').input_ids.to(self.device)
        encode_ids = self.tokenizer(GUIDANCE_PROMPT, return_tensors='pt').input_ids.to(self.device)
        c = self.config
        Abstract_StreamKV.__init__(
            self, _ProcessorBridge(processor), n_frame_tokens, init_ids, None, encode_ids,
            c.compression_ratio, c.compress_temp, c.retrieve_temp, c.max_segment_size,
            c.min_segment_size, c.segment_theta, c.segment_mode, str(c.use_segment_summary),
            c.n_local, c.retrieve_size, c.retrieve_chunk_size)
        self.attention_settings = dict(
            n_init=init_ids.shape[1] if n_init is None else n_init,
            n_encode=encode_ids.shape[1] if n_encode is None else n_encode,
            use_segment_summary=str(c.use_segment_summary), n_local=c.n_local,
            compression_ratio=c.compression_ratio, retrieve_local=str(c.retrieve_local),
            retrieve_local_size=c.retrieve_local_size, fattn=fattn,
            block_size=n_frame_tokens, retrieve_size=c.retrieve_size,
            chunk_size=c.retrieve_chunk_size, max_cached_block=128,
            exc_block_size=n_frame_tokens, pin_memory=pin_memory,
            async_global_stream=async_global_stream)
        # The reference rebuilds unscaled ESM frequencies from rope_theta. Pass
        # base explicitly so the shared bridge does not copy scaled HF inv_freq.
        patch_hf(self.decoder, attention_factory=streamkv_attention_forward,
                 base=self.decoder.config.rope_parameters['rope_theta'], distance_scale=1.,
                 **self.attention_settings)
        self.kv_cache = None

    @classmethod
    def load(cls, model_path, config=None):
        from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor
        processor = LlavaOnevisionProcessor.from_pretrained(model_path)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, device_map='auto', low_cpu_mem_usage=True, dtype=torch.float16)
        return cls(model, processor, config)

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    @property
    def dtype(self):
        return self.model.dtype

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_prompt(self, question, mc=False):
        return question_prompt('onevision', question, mc)

    def _get_video_features(self, pixel_values_videos):
        # Same operations/order as the reference adapter; only attribute paths
        # moved under .model in Transformers 5.
        batch_size, frames, channels, height, width = pixel_values_videos.shape
        pixels = pixel_values_videos.view(batch_size * frames, channels, height, width)
        output = self.model.model.vision_tower(pixels, output_hidden_states=True)
        features = output.hidden_states[self.model.config.vision_feature_layer]
        if self.model.config.vision_feature_select_strategy == 'default':
            features = features[:, 1:]
        features = self.model.model.multi_modal_projector(features)
        features = self.model.model.apply_pooling(features)
        return features.reshape(batch_size, frames * features.shape[1], -1)

    def close(self):
        self.clear_cache()
