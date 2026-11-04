"""CUDA segment scoring and stable, batched headwise partitioning.

Imported lazily. The reference implementations remain in segments/cache.
No full QK logits or FP32 copy of the segment's keys is written to memory.
"""
from contextlib import nullcontext

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N', 'D', 'SH', 'ST', 'SD'])
def _frame_query_mean(Q, OUT, N, D, SH, ST, SD, BT: tl.constexpr, BD: tl.constexpr):
    head, block = tl.program_id(0), tl.program_id(1)
    d = block * BD + tl.arange(0, BD)
    total = tl.full((BD,), 0., tl.float32)
    for start in range(tl.cdiv(N, BT)):
        t = start * BT + tl.arange(0, BT)
        q = tl.load(Q + head * SH + t[:, None] * ST + d[None, :] * SD,
                    (t[:, None] < N) & (d[None, :] < D), 0).to(tl.float32)
        total += tl.sum(q, 0)
    # Nonfinite inputs propagate to the existing segment-commit validation.
    # Do not add a host synchronization to every encoded frame and layer.
    tl.store(OUT + head * D + d, total / N, d < D)


def frame_query_mean(query):
    """Fuse cast and token reduction, without materializing FP32 [H, N, D]."""
    heads, tokens, dim = query.shape
    out = torch.empty((heads, dim), dtype=torch.float32, device=query.device)
    context = torch.cuda.device(query.device) if query.is_cuda else nullcontext()
    with context:
        _frame_query_mean[(heads, triton.cdiv(dim, 16))](
            query, out, tokens, dim, *query.stride(), min(256, triton.next_power_of_2(tokens)), 16,
            enable_fp_fusion=False)
    return out


@triton.jit(do_not_specialize=['F', 'N'])
def _score(Q, K, OUT, ERROR, F, N, H: tl.constexpr, G: tl.constexpr,
           D: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    block, h, frame = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = block * BN + tl.arange(0, BN)
    d = tl.arange(0, BD)
    # K is [key frame, KV head, token, channel], in its original dtype.
    k = tl.load(K + ((frame * H + h) * N + t[None, :]) * D + d[:, None],
                (t[None, :] < N) & (d[:, None] < D), 0).to(tl.float32)
    invalid = tl.sum(tl.sum((tl.abs(k) == float('inf')) | (k != k), 0), 0) > 0
    total = tl.full((BN,), 0., tl.float32)
    for g in range(G):
        frame_sum = tl.full((BN,), 0., tl.float32)
        for start in range(tl.cdiv(F, BQ)):
            f = start * BQ + tl.arange(0, BQ)
            q = tl.load(Q + ((f[:, None] * H + h) * G + g) * D + d[None, :],
                        (f[:, None] < F) & (d[None, :] < D), 0)
            invalid |= tl.sum(tl.sum((tl.abs(q) == float('inf')) | (q != q), 0), 0) > 0
            # Keep every frame query/GQA head separate until AFTER its dot.
            logits = tl.dot(q, k, input_precision='ieee') / tl.sqrt(float(D))
            frame_sum += tl.sum(tl.where(f[:, None] < F, logits, 0.), 0)
        total += frame_sum / F
    scores = total / G
    invalid |= tl.sum(((tl.abs(scores) == float('inf')) | (scores != scores)) & (t < N), 0) > 0
    if invalid:
        tl.atomic_or(ERROR, 1)
    tl.store(OUT + (frame * H + h) * N + t, scores, t < N)


@triton.jit(do_not_specialize=['N'])
def _partition(SCORES, RANKS, COUNTS, N, H: tl.constexpr, B: tl.constexpr):
    frame, h = tl.program_id(0), tl.program_id(1)
    t = tl.arange(0, B)
    row = frame * H + h
    scores = tl.load(SCORES + row * N + t, t < N, 0.)
    mean = tl.sum(scores, 0) / N
    high = (t < N) & (scores >= mean)
    prefix = tl.cumsum(high.to(tl.int32), 0)
    # Nonnegative ranks address high; -(rank + 1) addresses low.
    rank = tl.where(high, prefix - 1, -(t - prefix + 1))
    tl.store(RANKS + row * N + t, rank, t < N)
    invalid = tl.sum(((scores != scores) | (tl.abs(scores) == float('inf'))) & (t < N), 0) > 0
    tl.store(COUNTS + row, tl.where(invalid, -1, tl.sum(high.to(tl.int32), 0)))


@triton.jit(do_not_specialize=['N'])
def _scatter(DESC, RANKS, N, H: tl.constexpr, D: tl.constexpr,
             DTYPE: tl.constexpr, B: tl.constexpr):
    frame = tl.program_id(1)
    desc = DESC + frame * (9 + 2 * H)
    i = tl.program_id(0) * B + tl.arange(0, B)
    row, d = i // D, i % D
    h, t = row // N, row % N
    valid = row < H * N
    rank = tl.load(RANKS + frame * H * N + row, valid, 0)
    high = rank >= 0
    offset = tl.load(desc + 9 + tl.where(high, h, H + h), valid, 0)
    target = offset + tl.where(high, rank, -rank - 1)
    k = tl.load(tl.load(desc).to(tl.pointer_type(DTYPE)) + i, valid, 0)
    v = tl.load(tl.load(desc + 1).to(tl.pointer_type(DTYPE)) + i, valid, 0)
    kp = tl.where(high, tl.load(desc + 3), tl.load(desc + 6)).to(tl.pointer_type(DTYPE))
    vp = tl.where(high, tl.load(desc + 4), tl.load(desc + 7)).to(tl.pointer_type(DTYPE))
    tl.store(kp + target * D + d, k, valid)
    tl.store(vp + target * D + d, v, valid)
    ids = tl.load(tl.load(desc + 2).to(tl.pointer_type(tl.int32)) + t, valid & (d == 0), 0)
    ip = tl.where(high, tl.load(desc + 5), tl.load(desc + 8)).to(tl.pointer_type(tl.int32))
    tl.store(ip + target, ids, valid & (d == 0))


def frame_query_scores(query, keys):
    """Validated, same-shaped keys; return FP32 [frames, heads, tokens]."""
    frames, (heads, tokens, dim) = len(keys), keys[0].shape
    query = query.to(device=keys[0].device, dtype=torch.float32).contiguous()
    keys = torch.stack(keys)
    scores = torch.empty((frames, heads, tokens), device=keys.device, dtype=torch.float32)
    error = torch.zeros((), device=keys.device, dtype=torch.int32)
    context = torch.cuda.device(keys.device) if keys.is_cuda else nullcontext()
    with context:
        _score[(triton.cdiv(tokens, 32), heads, frames)](
            query, keys, scores, error, frames, tokens, heads, query.shape[1] // heads,
            dim, 32, 32, max(32, triton.next_power_of_2(dim)),
            enable_fp_fusion=False, num_warps=4, num_stages=1)
    if error.item():
        raise ValueError('Nonfinite frame Q/K or attention scores')
    return scores


def partition_frames(entries, scores):
    """One counts readback and one scatter launch for every frame in a layer.

    Each output region owns exact-sized storage, so dropping a low region or
    evicting one frame really releases its payload even while others survive.
    """
    frames, heads, tokens = scores.shape
    dim, device, dtype = entries[0].high_k.shape[-1], scores.device, entries[0].high_k.dtype
    scores = scores.contiguous()
    ranks = torch.empty(scores.shape, dtype=torch.int32, device=device)
    counts = torch.empty((frames, heads), dtype=torch.int32, device=device)
    context = torch.cuda.device(device) if scores.is_cuda else nullcontext()
    with context:
        _partition[(frames, heads)](scores, ranks, counts, tokens, heads, triton.next_power_of_2(tokens))
    host_counts = counts.tolist()  # Single readback for the entire layer.
    if any(n < 0 for row in host_counts for n in row):
        raise ValueError('Nonfinite frame attention scores')
    outputs, descriptors = [], []
    for entry, row in zip(entries, host_counts):
        high_offsets, low_offsets = [0], [0]
        for n in row:
            high_offsets.append(high_offsets[-1] + n)
            low_offsets.append(low_offsets[-1] + tokens - n)
        hk = torch.empty((high_offsets[-1], dim), dtype=dtype, device=device)
        hv = torch.empty_like(hk)
        hi = torch.empty(high_offsets[-1], dtype=torch.int32, device=device)
        lk = torch.empty((low_offsets[-1], dim), dtype=dtype, device=device)
        lv = torch.empty_like(lk)
        li = torch.empty(low_offsets[-1], dtype=torch.int32, device=device)
        descriptors.append([x.data_ptr() for x in
                            (entry.high_k, entry.high_v, entry.high_indices, hk, hv, hi, lk, lv, li)]
                           + high_offsets[:-1] + low_offsets[:-1])
        outputs.append(dict(high_k=hk, high_v=hv, high_indices=hi,
                            low_k=lk, low_v=lv, low_indices=li,
                            high_offsets=tuple(high_offsets), low_offsets=tuple(low_offsets)))
    desc = torch.tensor(descriptors, dtype=torch.int64, device=device)
    scalar_type = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}[dtype]
    context = torch.cuda.device(device) if scores.is_cuda else nullcontext()
    with context:
        _scatter[(triton.cdiv(heads * tokens * dim, 256), frames)](
            desc, ranks, tokens, heads, dim, scalar_type, 256)
    return outputs
