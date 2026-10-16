"""Window attention and conditional entropy; no KV pruning or pooling."""
import math

import numpy as np
import torch


WINDOW_SIZE = 32


def window_importance(query, key, *, scaling=None, window=WINDOW_SIZE, attention_mask=None):
    """Mean attention over heads/last W queries, with the FULL causal denominator.

    Inputs are post-RoPE Q/K, [1, heads, sequence, head_dim]. GQA heads are
    matched before aggregation. Only W x N weights per head group are materialized.
    Window positions are returned too, to audit conservation of probability mass.
    """
    if query.shape[0] != 1 or key.shape[0] != 1:
        raise ValueError("Observation requires batch size 1")
    length = query.shape[-2]
    if length != key.shape[-2] or length <= window:
        raise ValueError("Observation requires a full prefill longer than the window")
    heads, kv_heads = query.shape[1], key.shape[1]
    if heads % kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    positions = torch.arange(length, device=query.device)
    causal = positions[None, :] <= positions[-window:, None]
    total = torch.zeros(length, dtype=torch.float32, device=query.device)
    for start in range(0, heads, 4):
        stop = min(heads, start + 4)
        kv_indices = torch.arange(start, stop, device=key.device) // (heads // kv_heads)
        q = query[0, start:stop, -window:].float()
        k = key[0, kv_indices].float()
        logits = (q @ k.transpose(-1, -2)) * scale
        logits.masked_fill_(~causal, -torch.inf)
        if attention_mask is not None:
            mask = attention_mask[..., -window:, :length]
            if mask.shape[1] != 1:
                mask = mask[:, start:stop]
            mask = mask[0]
            if mask.dtype == torch.bool:
                logits.masked_fill_(~mask, -torch.inf)
            else:
                logits += mask.float()
        total += logits.softmax(dim=-1).sum(dim=(0, 1)) / (heads * window)
    if not torch.isfinite(total).all():
        raise ValueError("Non-finite observation attention")
    return total.cpu().numpy()


def frame_statistics(scores, frames):
    """H(frame) uses p_i = score_i / frame_mass, over historical tokens only."""
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Importance scores must be finite and nonnegative")
    result = []
    for frame in frames:
        start, end = frame['token_start'], frame['token_end']
        values = scores[start:min(end, len(scores))]
        mass, count = float(values.sum()), len(values)
        entropy = normalized = effective = None
        if count and mass > 0:
            p = values[values > 0] / mass
            entropy = max(0.0, float(-(p * np.log(p)).sum()))
            normalized = entropy / math.log(count) if count > 1 else 0.0
            effective = math.exp(entropy)
        result.append(dict(index=frame['index'], mass=mass, entropy=entropy,
                           normalized_entropy=normalized, effective_tokens=effective,
                           scored_tokens=count, window_tokens=end - start - count))
    return result


def build_views(layer_scores, frames, prefix_length):
    full = np.asarray(layer_scores, dtype=np.float64)
    if full.ndim != 2 or not np.allclose(full.sum(axis=1), 1, atol=2e-5):
        raise ValueError("Attention mass must sum to one in every layer")
    views = []
    for name, vector in [('all', full.mean(axis=0))] + [
        (str(index), score) for index, score in enumerate(full)
    ]:
        scores = vector[:prefix_length]
        stats = frame_statistics(scores, frames)
        video_mass = sum(frame['mass'] for frame in stats)
        views.append(dict(id=name, scores=scores.tolist(), frames=stats,
                          history_mass=float(scores.sum()), video_mass=video_mass,
                          text_mass=max(0.0, float(scores.sum()) - video_mass),
                          window_mass=float(vector[prefix_length:].sum())))
    return views
