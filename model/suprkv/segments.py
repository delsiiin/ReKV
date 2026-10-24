"""Incremental equivalent of observation's historical-EMA segmenter."""
import math

import torch

from observation.segments import EPSILON, vision_frame_distances


class OnlineSegmenter:
    def __init__(self, alpha=.2, min_frames=1, max_frames=32):
        if not 0 < alpha <= 1 or not 1 <= min_frames <= max_frames:
            raise ValueError('Invalid segment limits or EMA alpha')
        self.alpha, self.min_frames, self.max_frames = alpha, min_frames, max_frames
        self.previous = None
        self.ema = None
        self.count = 0

    def push(self, patches):
        """Return a boundary BEFORE this frame; caller closes at the cap AFTER it."""
        distances, self.previous = vision_frame_distances(patches[None], self.previous)
        boundary = False
        if len(distances):
            distance = float(distances[0])
            if not math.isfinite(distance):
                raise ValueError('Nonfinite visual distance')
            boundary = (self.ema is not None and distance > max(EPSILON, self.ema)
                        and self.count >= self.min_frames)
            self.ema = distance if self.ema is None else (1-self.alpha)*self.ema + self.alpha*distance
        if boundary:
            self.count = 0
        self.count += 1
        return boundary

    def close(self):
        self.count = 0  # Preserve EMA and the preceding visual frame.


def summary_scores(query, key):
    """FP32 scores; Q=[summary tokens, query heads, d], K=[kv heads, tokens, d]."""
    if query.ndim != 3 or key.ndim != 3 or not len(query):
        raise ValueError('Empty or malformed summary Q/K')
    heads, kv_heads, dim = query.shape[1], key.shape[0], key.shape[-1]
    if heads % kv_heads or query.shape[-1] != dim:
        raise ValueError('Invalid GQA head mapping')
    query = query.to(device=key.device, dtype=torch.float32)
    key = key.float()
    if not torch.isfinite(query).all() or not torch.isfinite(key).all():
        raise ValueError('Nonfinite summary Q/K')
    # Linearity gives exactly mean_{query,head}(Q K^T), without an S x N matrix.
    with torch.autocast(device_type=key.device.type, enabled=False):
        q = query.mean(0).reshape(kv_heads, heads // kv_heads, dim).mean(1)
        return torch.einsum('hd,hnd->n', q, key) / (kv_heads * math.sqrt(dim))


def segment_summary_scores(query, keys):
    """Score equally shaped frames in one layer, pooling its Query only once.

    Return FP32 [frames, tokens]; callers must threshold each frame separately.
    Only one layer's FP32 keys are materialized at a time by the cache caller.
    """
    if not keys or query.ndim != 3 or not len(query):
        raise ValueError('Empty or malformed summary Q/K')
    key = keys[0]
    if key.ndim != 3 or any(k.shape != key.shape or k.device != key.device for k in keys):
        raise ValueError('Scoring keys must have matching frame shapes and devices')
    kv_heads, _, dim = key.shape
    heads = query.shape[1]
    if kv_heads < 1 or dim < 1 or heads < 1 or heads % kv_heads or query.shape[-1] != dim:
        raise ValueError('Invalid GQA head mapping')
    query = query.to(device=key.device, dtype=torch.float32)
    batch = torch.stack(keys).float()
    # One host synchronization per layer instead of two per frame/layer.
    if not (torch.isfinite(query).all() & torch.isfinite(batch).all()):
        raise ValueError('Nonfinite summary Q/K')
    with torch.autocast(device_type=key.device.type, enabled=False):
        q = query.mean(0).reshape(kv_heads, heads // kv_heads, dim).mean(1)
        return torch.einsum('hd,fhnd->fn', q, batch) / (kv_heads * math.sqrt(dim))
