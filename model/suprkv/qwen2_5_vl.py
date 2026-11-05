"""Qwen2.5-VL frontend for the shared SuPr-KV streaming/cache lifecycle."""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from model.rekv.qwen2_5_vl_vision import FRAME_GRID, FRAME_TOKENS, frame_features
from observation.segments import SUMMARY_SYSTEM, summary_prompt
from .onevision import OneVisionEngine
from .summarizer import OneVisionSummarizer


class Qwen2_5_VLSummarizer(OneVisionSummarizer):
    """Reuse per-frame visual features with native spatial MRoPE for summaries.

    Independent sampled frames are represented as images, each enclosed by
    Qwen's vision markers. Text decoding continues from the native RoPE offset,
    which differs from the number of cached tokens for visual sequences.
    """
    def prompt_ids(self, frames):
        prefix = f'<|im_start|>system\n{SUMMARY_SYSTEM}<|im_end|>\n<|im_start|>user\n'
        suffix = f'\n{summary_prompt(frames)}<|im_end|>\n<|im_start|>assistant\n'
        return tuple(self.tokenizer(text, return_tensors='pt').input_ids.to(self.device)
                     for text in (prefix, suffix))

    def context_length(self, features, frames):
        return super().context_length(features, frames) + 2 * len(features)

    def _inputs_and_positions(self, features, frames):
        if not features or len(features) != len(frames):
            raise ValueError('Summary requires one feature per frame')
        prefix, suffix = self.prompt_ids(frames)
        config = self.model.config
        start = torch.tensor([[config.vision_start_token_id]], device=self.device)
        end = torch.tensor([[config.vision_end_token_id]], device=self.device)
        embed = self.decoder.get_input_embeddings()
        ids, inputs = [prefix], [embed(prefix)]
        for feature in features:
            if feature.shape[1] != FRAME_TOKENS:
                raise ValueError('Unexpected Qwen2.5-VL summary frame size')
            pad = torch.full((1, FRAME_TOKENS), config.image_token_id, device=self.device)
            ids.extend((start, pad, end))
            inputs.extend((embed(start), feature.to(self.device, self.dtype), embed(end)))
        ids.append(suffix)
        inputs.append(embed(suffix))
        ids = torch.cat(ids, 1)
        modality = (ids == config.image_token_id).long()
        side = FRAME_GRID * config.vision_config.spatial_merge_size
        grid = torch.tensor([[1, side, side]] * len(features), device=self.device)
        positions, _ = self.model.model.get_rope_index(
            ids, mm_token_type_ids=modality, image_grid_thw=grid)
        return torch.cat(inputs, 1), positions

    @torch.inference_mode()
    def generate(self, features, frames, max_new_tokens, check_cancel):
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        check_cancel()
        with self.decoder_context():
            inputs, positions = self._inputs_and_positions(features, frames)
            output = self.decoder(inputs_embeds=inputs, position_ids=positions,
                                  use_cache=True, return_dict=True)
            next_position = int(positions.max()) + 1
            tokens = []
            for step in range(max_new_tokens):
                check_cancel()
                token = int(self.model.lm_head(output.last_hidden_state[:, -1:])[0, -1].argmax())
                tokens.append(token)
                if token == self.tokenizer.eos_token_id or step + 1 == max_new_tokens:
                    break
                output = self.decoder(
                    input_ids=torch.tensor([[token]], device=self.device),
                    position_ids=torch.full((3, 1, 1), next_position + step, device=self.device),
                    past_key_values=output.past_key_values, use_cache=True, return_dict=True)
        return self.tokenizer.decode(tokens, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False).strip()


class Qwen2_5_VLEngine(OneVisionEngine):
    family = 'qwen2_5_vl'
    frame_tokens = FRAME_TOKENS
    model_class = Qwen2_5_VLForConditionalGeneration
    processor_class = Qwen2_5_VLProcessor
    summarizer_class = Qwen2_5_VLSummarizer

    def _frame_features(self, image):
        feature = frame_features(self.model, self.processor, torch.as_tensor(image)[None])
        # Use merged visual embeddings for segmentation; Qwen does not expose
        # OneVision's selected CLIP/SigLIP hidden states.
        return feature, feature[0]
