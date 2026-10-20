"""Window attention and conditional entropy; no KV pruning or pooling."""
import math

import numpy as np
import torch


WINDOW_SIZE = 32


def query_importance(query, key, query_positions, *, scaling=None, attention_mask=None,
                     query_chunk_size=32, check_cancel=lambda: None):
    """Mean attention for absolute query positions, with the full causal denominator.

    Post-RoPE Q/K are [1, heads, sequence, head_dim]. K includes the native
    cache; Q is its final contiguous suffix. Only chunk_size x keys weights
    per head group are materialized, even when observing all video tokens.
    """
    if query.shape[0] != 1 or key.shape[0] != 1:
        raise ValueError("Observation requires batch size 1")
    length = key.shape[-2]
    offset = length - query.shape[-2]
    selected = torch.as_tensor(query_positions, device=query.device, dtype=torch.long)
    if (offset < 0 or selected.ndim != 1 or not selected.numel()
            or (selected < offset).any() or (selected >= length).any()):
        raise ValueError("Query positions must select available queries in the current forward")
    if type(query_chunk_size) is not int or query_chunk_size < 1:
        raise ValueError("query_chunk_size must be a positive integer")
    heads, kv_heads = query.shape[1], key.shape[1]
    if heads % kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    positions = torch.arange(length, device=query.device)
    total = torch.zeros(length, dtype=torch.float32, device=query.device)
    for start in range(0, heads, 4):
        stop = min(heads, start + 4)
        kv_indices = torch.arange(start, stop, device=key.device) // (heads // kv_heads)
        k = key[0, kv_indices].float()
        for chunk in selected.split(query_chunk_size):
            check_cancel()
            local = chunk - offset
            q = query[0, start:stop].index_select(-2, local).float()
            logits = (q @ k.transpose(-1, -2)) * scale
            logits.masked_fill_(positions[None, :] > chunk[:, None], -torch.inf)
            if attention_mask is not None:
                mask = attention_mask[..., :length]
                if mask.shape[-2] != 1:
                    mask = mask.index_select(-2, local)
                if mask.shape[1] != 1:
                    mask = mask[:, start:stop]
                mask = mask[0]
                if mask.dtype == torch.bool:
                    logits.masked_fill_(~mask, -torch.inf)
                else:
                    logits += mask.float()
            total += logits.softmax(dim=-1).sum(dim=(0, 1)) / (heads * selected.numel())
    if not torch.isfinite(total).all():
        raise ValueError("Non-finite observation attention")
    return total.cpu().numpy()


def window_importance(query, key, *, scaling=None, window=WINDOW_SIZE, attention_mask=None):
    """Compatibility wrapper for the last W queries of a complete prefill."""
    length = query.shape[-2]
    if type(window) is not int or window < 1 or length != key.shape[-2] or length <= window:
        raise ValueError("Observation requires a positive window smaller than a full prefill")
    return query_importance(query, key, range(length - window, length),
                            scaling=scaling, attention_mask=attention_mask)


def validate_video_window(window, video_tokens):
    if type(window) is not int or not 1 <= window <= video_tokens:
        raise ValueError(f"video_window_size must be an integer from 1 to {video_tokens} "
                         "(the actual number of video tokens)")
    return window


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


def build_comparison(before_scores, after_scores, frames, queries):
    """Keep identical, complete video key positions in both phases."""
    positions = [i for frame in frames for i in range(frame['token_start'], frame['token_end'])]
    phase_views = {}
    for phase, scores in [('before', before_scores), ('after', after_scores)]:
        full = np.asarray(scores, dtype=np.float64)
        if (full.ndim != 2 or not np.isfinite(full).all() or (full < 0).any()
                or not np.allclose(full.sum(axis=1), 1, atol=2e-5)):
            raise ValueError("Comparison attention must be finite, nonnegative and sum to one")
        views = []
        for vector in [full.mean(axis=0), *full]:
            stats = frame_statistics(vector, frames)
            video_mass = float(vector[positions].sum())
            views.append(dict(scores=vector[positions].tolist(), frames=stats,
                              video_mass=video_mass,
                              text_mass=max(0.0, float(vector.sum()) - video_mass)))
        phase_views[phase] = views
    if len(phase_views['before']) != len(phase_views['after']):
        raise ValueError("Comparison phases must have identical layers")
    return dict(
        queries=queries, token_positions=positions,
        method=dict(aggregation='mean over selected queries and attention heads; all = mean over layers',
                    denominator='all causally visible keys, including text and query-window keys',
                    video_keys='all video tokens, including the video query window',
                    difference='after - before'),
        views=[dict(id='all' if i == 0 else str(i - 1), before=before, after=after)
               for i, (before, after) in enumerate(zip(phase_views['before'], phase_views['after']))],
    )
