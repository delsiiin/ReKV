"""Head-batched encoding using ReKV's existing online-softmax inner loop."""
import math

import torch
import triton
import triton.language as tl

from model.rekv.attention.dot_production_attention import triton_impl as legacy
from model.rekv.attention.dot_production_attention.triton_impl import _attn_fwd_inner


@triton.jit
def _copy_segments(DESC, KOUT, VOUT, D: tl.constexpr, BT: tl.constexpr):
    row = DESC+tl.program_id(1)*8
    kp, vp = tl.load(row).to(KOUT.dtype), tl.load(row+1).to(VOUT.dtype)
    n, offset = tl.load(row+2), tl.load(row+3)
    kt, kd, vt, vd = tl.load(row+4), tl.load(row+5), tl.load(row+6), tl.load(row+7)
    t = tl.program_id(0)*BT+tl.arange(0, BT)
    d = tl.arange(0, D)
    k = tl.load(kp+t[:, None]*kt+d[None, :]*kd, t[:, None]<n, 0)
    v = tl.load(vp+t[:, None]*vt+d[None, :]*vd, t[:, None]<n, 0)
    out = (offset+t[:, None])*D+d[None, :]
    tl.store(KOUT+out, k, t[:, None]<n)
    tl.store(VOUT+out, v, t[:, None]<n)


@triton.jit(do_not_specialize=['M'])
def _pack(DESC, K, V, KOUT, VOUT, M, D: tl.constexpr, BT: tl.constexpr):
    head = tl.program_id(1)
    row = DESC + head * 8
    kp, vp = tl.load(row), tl.load(row + 1)
    past, offset = tl.load(row + 2), tl.load(row + 3)
    kt, kd, vt, vd = tl.load(row + 4), tl.load(row + 5), tl.load(row + 6), tl.load(row + 7)
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    history = t[:, None] < past
    tail = (t[:, None] >= past) & (t[:, None] < past + M)
    pk = tl.load(kp.to(K.dtype) + t[:, None] * kt + d[None, :] * kd, history, 0)
    pv = tl.load(vp.to(V.dtype) + t[:, None] * vt + d[None, :] * vd, history, 0)
    current = (head * M + t[:, None] - past) * D + d[None, :]
    ck, cv = tl.load(K + current, tail, 0), tl.load(V + current, tail, 0)
    destination = (offset + t[:, None]) * D + d[None, :]
    tl.store(KOUT + destination, tl.where(history, pk, ck), history | tail)
    tl.store(VOUT + destination, tl.where(history, pv, cv), history | tail)


@triton.jit(do_not_specialize=['M'])
def _attend(Q, K, V, DESC, OUT, M, sm_scale, G: tl.constexpr, D: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr):
    block, head = tl.program_id(0), tl.program_id(1)
    row = DESC + (head // G) * 8
    past, offset = tl.load(row + 2).to(tl.int32), tl.load(row + 3)
    total = past + M
    qp = tl.make_block_ptr(Q + head * M * D, (M, D), (D, 1), (block * BM, 0), (BM, D), (1, 0))
    kp = tl.make_block_ptr(K + offset * D, (D, total), (1, D), (0, 0), (D, BN), (0, 1))
    vp = tl.make_block_ptr(V + offset * D, (total, D), (D, 1), (0, 0), (BN, D), (1, 0))
    op = tl.make_block_ptr(OUT + head * M * D, (M, D), (D, 1), (block * BM, 0), (BM, D), (1, 0))
    q = tl.load(qp, boundary_check=(0, 1), padding_option='zero')
    acc = tl.zeros((BM, D), tl.float32)
    maximum = tl.full((BM,), float('-inf'), tl.float32)
    denominator = tl.full((BM,), 1., tl.float32)
    scale = sm_scale
    scale *= 1.4426950408889634
    # Sliding offset=past, window=total is exactly full-prefix causal attention.
    # The same tile order and accumulation as the per-head reference are used.
    acc, denominator, maximum = _attn_fwd_inner(
        acc, denominator, maximum, q, kp, vp, block, scale, total, past, total,
        BM, D, BN, True, False, False, False)
    acc = acc / denominator[:, None]
    tl.store(op, acc.to(OUT.dtype.element_ty), boundary_check=(0, 1))


def pack_and_attend(query, key, value, descriptors, tokens, longest, copies=None):
    _, heads, count, dim = query.shape
    desc = torch.tensor(descriptors, dtype=torch.int64, device=query.device)
    packed_k = torch.empty((tokens, dim), dtype=query.dtype, device=query.device)
    packed_v = torch.empty_like(packed_k)
    out = torch.empty_like(query)
    with torch.cuda.device(query.device):
        if copies is None:
            _pack[(triton.cdiv(longest, 32), len(descriptors))](desc, key, value, packed_k, packed_v, count, dim, 32)
        else:
            copy_desc = torch.tensor(copies, dtype=torch.int64, device=query.device)
            _copy_segments[(triton.cdiv(longest, 32), len(copies))](copy_desc, packed_k, packed_v, dim, 32)
        bm, bn = legacy._BLOCK_M, legacy._BLOCK_N
        def run(block_n):
            _attend[(triton.cdiv(count, bm), heads)](
                query, packed_k, packed_v, desc, out, count, 1 / math.sqrt(dim),
                heads // len(descriptors), dim, bm, block_n)
        try:
            run(bn)
        except triton.OutOfResources:
            if bn <= 32:
                raise
            run(32)
    return out
