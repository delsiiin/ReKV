"""Full-context inference with the original, unpatched language model.

Only visual embeddings persist across questions. Each question starts with the
entire visible prefix and gets a fresh native KV cache, discarded after decoding.
"""
import torch

from model.video_backbone import INITIAL_PROMPTS, question_prompt, preprocess_video, video_features


class NativeRuntime:
    def __init__(self, model, processor, family, max_new_tokens=None):
        self.model, self.processor, self.family = model, processor, family
        self.tokenizer = processor.tokenizer
        self.decoder = model.model if family == 'longva' else model.model.language_model
        self.max_new_tokens = max_new_tokens
        self.reset()

    @classmethod
    def load(cls, model_name, model_path, *, device=None, max_new_tokens=None):
        # Import only the selected original backbone. Never import a ReKV loader or patch.
        options = dict(device_map=device or 'auto', low_cpu_mem_usage=True,
                       dtype=torch.float32 if device == 'cpu' else torch.float16)
        if model_name.startswith('llava_ov_'):
            from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor
            model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_path, **options)
            processor = LlavaOnevisionProcessor.from_pretrained(model_path)
            family = 'onevision'
        elif model_name == 'video_llava_7b':
            from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
            model = VideoLlavaForConditionalGeneration.from_pretrained(model_path, **options)
            processor = VideoLlavaProcessor.from_pretrained(model_path)
            family = 'video_llava'
        elif model_name == 'longva_7b':
            from transformers import AutoTokenizer
            from longva.model import LlavaQwenForCausalLM
            model = LlavaQwenForCausalLM.from_pretrained(model_path, **options)
            tower = model.get_vision_tower()
            if not tower.is_loaded:
                tower.load_model(device_map=device or 'auto')
            processor = tower.image_processor
            processor.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            family = 'longva'
        else:
            raise ValueError(f'Unsupported native model: {model_name}')
        return cls(model.eval(), processor, family, max_new_tokens)

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    def reset(self):
        self.frames = []

    def close(self):
        self.reset()

    @torch.inference_mode()
    def append_frame(self, frame, timestamp):
        frames = torch.from_numpy(frame).unsqueeze(0)
        pixels = preprocess_video(self.processor, self.family, frames)
        pixels = pixels.to(device=self.device, dtype=self.model.dtype)
        features = video_features(self.model, self.family, pixels)
        self.frames.append(features.to(self.device))

    def get_prompt(self, question, mc=False):
        return question_prompt(self.family, question, mc)

    def _embed_text(self, text, *, continuation=False):
        ids = self.tokenizer(text, return_tensors='pt').input_ids.to(self.device)
        # Video-LLaVA's tokenizer inserts BOS; it belongs only at the start.
        if continuation and self.family == 'video_llava':
            ids = ids[:, 1:]
        return self.model.get_input_embeddings()(ids)

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128):
        limit = self.max_new_tokens if self.max_new_tokens is not None else max_new_tokens
        if limit < 1:
            raise ValueError('max_new_tokens must be positive')
        prefix = self._embed_text(INITIAL_PROMPTS[self.family])
        question = self._embed_text(input_text['prompt'], continuation=True)
        inputs = torch.cat([prefix, *self.frames, question], dim=1)
        output = self.decoder(inputs_embeds=inputs, use_cache=True, return_dict=True)
        tokens = []
        stop_ids = self.tokenizer.eos_token_id
        stop_ids = set(stop_ids if isinstance(stop_ids, (list, tuple)) else [stop_ids])
        for index in range(limit):
            # Apply the head only to the last token, including during prefill.
            logits = self.model.lm_head(output.last_hidden_state[:, -1:, :])
            token = int(logits[0, -1].argmax())
            tokens.append(token)
            if token in stop_ids or index == limit - 1:
                break
            output = self.decoder(input_ids=torch.tensor([[token]], device=self.device),
                                  past_key_values=output.past_key_values,
                                  use_cache=True, return_dict=True)
        return self.tokenizer.decode(tokens, skip_special_tokens=True,
                                     spaces_between_special_tokens=False,
                                     clean_up_tokenization_spaces=True)
