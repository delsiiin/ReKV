"""Original Qwen2.5-VL processor and generation, without retrieval patches."""
import math

import torch

from model.video_backbone import INITIAL_PROMPTS
from .runtime import NativeRuntime


class Qwen2_5_VLNativeRuntime(NativeRuntime):
    """Keep sampled RGB frames; process the visible video afresh for each question.

    Native temporal patches can span adjacent frames. Reprocessing the visible
    prefix also handles odd frame counts without retaining a padded last patch
    after the next frame arrives. The checkpoint controls spatial resolution.
    """
    def __init__(self, model, processor, family='qwen2_5_vl', max_new_tokens=None, *, sample_fps=None):
        if sample_fps is not None and (not math.isfinite(sample_fps) or sample_fps <= 0):
            raise ValueError('sample_fps must be finite and positive')
        self.sample_fps = sample_fps
        super().__init__(model, processor, family, max_new_tokens)

    def reset(self):
        self.frames = []
        self.timestamps = []
        self.model.model.rope_deltas = None

    def append_frame(self, frame, timestamp):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or (self.timestamps and timestamp <= self.timestamps[-1]):
            raise ValueError('Frame timestamps must be finite and strictly increasing')
        frame = torch.as_tensor(frame)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError('Expected an RGB frame with shape (H, W, 3)')
        if self.frames and frame.shape != self.frames[0].shape:
            raise ValueError('Frames in a video must have the same shape')
        # The sampler may reuse its buffer. Keep owned CPU pixels, not GPU KV.
        self.frames.append(frame.detach().cpu().clone())
        self.timestamps.append(timestamp)

    def _prepare_inputs(self, prompt):
        prefix = INITIAL_PROMPTS[self.family]
        if self.frames:
            count = len(self.frames)
            # The shared sampler supplies source timestamps. Convert their mean
            # interval to sampled FPS; for one frame use the requested sample
            # rate, or 1 FPS when a direct Python caller supplied no rate.
            fps = ((count - 1) / (self.timestamps[-1] - self.timestamps[0])
                   if count > 1 else self.sample_fps or 1.0)
            metadata = [dict(total_num_frames=count, fps=fps, frames_indices=list(range(count)))]
            inputs = self.processor(
                text=[prefix + '<|video_pad|>' + prompt],
                videos=[torch.stack(self.frames)], video_metadata=metadata,
                input_data_format='channels_last', do_sample_frames=False,
                return_tensors='pt')
        else:
            # A question before the first visible frame is a text-only request.
            text = prefix.removesuffix('<|vision_start|>') + prompt.removeprefix('<|vision_end|>')
            inputs = self.processor(text=[text], return_tensors='pt')
        return inputs.to(device=self.device, dtype=self.model.dtype)

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128):
        limit = self.max_new_tokens if self.max_new_tokens is not None else max_new_tokens
        if limit < 1:
            raise ValueError('max_new_tokens must be positive')
        inputs = self._prepare_inputs(input_text['prompt'])
        # Qwen keeps RoPE deltas on the model. Each question is an independent
        # native generation, including the text-only case after reset().
        self.model.model.rope_deltas = None
        try:
            output = self.model.generate(
                **inputs, max_new_tokens=limit, do_sample=False, num_beams=1,
                use_cache=True, return_dict_in_generate=False)
        finally:
            self.model.model.rope_deltas = None
        tokens = output[0, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(tokens, skip_special_tokens=True,
                                     spaces_between_special_tokens=False,
                                     clean_up_tokenization_spaces=True)
