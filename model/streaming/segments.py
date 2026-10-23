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
    """Q=[summary tokens, query heads, d], K=[kv heads, frame tokens, d]."""
    if query.ndim != 3 or key.ndim != 3 or not len(query):
        raise ValueError('Empty or malformed summary Q/K')
    heads, kv_heads, dim = query.shape[1], key.shape[0], key.shape[-1]
    if heads % kv_heads or query.shape[-1] != dim:
        raise ValueError('Invalid GQA head mapping')
    if not torch.isfinite(query).all() or not torch.isfinite(key).all():
        raise ValueError('Nonfinite summary Q/K')
    # Linearity gives exactly mean_{query,head}(Q K^T), without an S x N matrix.
    q = query.to(device=key.device, dtype=torch.float64).mean(0)
    q = q.reshape(kv_heads, heads // kv_heads, dim).mean(1)
    return torch.einsum('hd,hnd->n', q, key.double()) / (kv_heads * math.sqrt(dim))
