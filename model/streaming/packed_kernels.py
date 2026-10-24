"""CUDA kernels for PM-KVQ's MSB-first INT16 containers.

Like KIVI's quant/csrc/gemv_cuda.cu, decode tiles at the point of use;
never write an unpacked KV matrix to global memory before multiplication.
This implementation uses Triton and the resident cache's own layout/affine
convention, including FP32 parameters and rounding to the model dtype.
Imported lazily so CPU/reference operation does not require Triton.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _lower_codes(code, N: tl.constexpr):
    return (((code << (2*N)) - (code << N) + code
             + (2**(3*N-1)-2**(2*N-1)+2**(N-1))) >> (3*N)) & ((1 << N)-1)


# Retained token counts differ by frame AND layer. Specializing these sizes
# produces hundreds of binaries during a single observation run. Keep only
# the quantization format/tile shape static; suppress alignment/==1 variants
# for all variable sizes and strides as well.
@triton.jit(do_not_specialize=['LENGTH', 'PAD', 'ROWS', 'GROUPS', 'SH', 'SR', 'SC'])
def _quantize(X, P, S, Z, LENGTH, PAD, ROWS, GROUPS, GROUP: tl.constexpr,
              SH, SR, SC, BITS: tl.constexpr, BLOCK: tl.constexpr):
    row, g = tl.program_id(0), tl.program_id(1)
    i = tl.arange(0, BLOCK)
    col = g * GROUP + i
    valid = (i < GROUP) & (col < LENGTH)
    x = tl.load(X + row // ROWS * SH + row % ROWS * SR + col * SC, valid, 0).to(tl.float32)
    lo = tl.min(tl.where(valid, x, float('inf')), 0)
    hi = tl.max(tl.where(valid, x, float('-inf')), 0)
    scale = tl.minimum(tl.maximum((hi-lo)/65535., 1.e-5), 1.e5)
    code = libdevice.nearbyint(tl.div_rn(x-lo, scale))
    code = tl.minimum(tl.maximum(code, 0.), 65535.).to(tl.int32)
    code = tl.where(valid, code, 0)
    # Preserve PM-KVQ's exact sequential integer arithmetic, including int32
    # wraparound, without materializing an intermediate INT16 cache.
    if BITS <= 8:
        code = _lower_codes(code, 8)
        scale = scale * 257.
    if BITS <= 4:
        code = _lower_codes(code, 4)
        scale = scale * 17.
    if BITS <= 2:
        code = _lower_codes(code, 2)
        scale = scale * 5.
    PACK: tl.constexpr = 16 // BITS
    lanes = tl.arange(0, PACK)
    codes = tl.reshape(code, (BLOCK // PACK, PACK))
    word = tl.sum(codes << ((PACK-1-lanes[None, :]) * BITS), 1)
    wi = tl.arange(0, BLOCK // PACK)
    wc = g * (GROUP // PACK) + wi
    tl.store(P + row * (PAD // PACK) + wc, word, (wi < GROUP // PACK) & (wc < PAD // PACK))
    tl.store(S + row * GROUPS + g, scale)
    tl.store(Z + row * GROUPS + g, -lo)


@triton.jit(do_not_specialize=['WORDS', 'PARAMS'])
def _shrink(P, S, O, OS, WORDS, PARAMS, BITS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    N: tl.constexpr = BITS // 2
    lanes = tl.arange(0, 16 // N)
    code_index = i[:, None] * (16 // N) + lanes[None, :]
    word = tl.load(P + code_index // (16 // BITS), i[:, None] < WORDS, 0).to(tl.int32)
    code = (word >> ((16 // BITS - 1 - code_index % (16 // BITS)) * BITS)) & ((1 << BITS)-1)
    code = _lower_codes(code, N)
    result = tl.sum(code << ((16 // N - 1 - lanes[None, :]) * N), 1)
    tl.store(O + i, result, i < WORDS)
    scale = tl.load(S + i, i < PARAMS, 0)
    tl.store(OS + i, scale * (2**N + 1), i < PARAMS)


def quantize(value, group_size, axis, bits=16):
    x = value.transpose(1, 2) if axis == 'channel' else value
    h, rows, length = x.shape
    pad = triton.cdiv(length, 8) * 8
    groups = triton.cdiv(length, group_size)
    data = torch.empty((h, rows, pad // (16 // bits)), device=x.device, dtype=torch.int16)
    scale = torch.empty((h, rows, groups), device=x.device, dtype=torch.float32)
    offset = torch.empty_like(scale)
    with torch.cuda.device(x.device):
        _quantize[(h*rows, groups)](x, data, scale, offset, length, pad, rows, groups,
                                   group_size, *x.stride(), bits, triton.next_power_of_2(group_size),
                                   enable_fp_fusion=False)
    return data, scale, offset


def shrink(packed):
    data = torch.empty((*packed.data.shape[:-1], packed.data.shape[-1] // 2),
                       device=packed.data.device, dtype=torch.int16)
    scale = torch.empty_like(packed.scale)
    with torch.cuda.device(data.device):
        _shrink[(triton.cdiv(max(data.numel(), scale.numel()), 128),)](
            packed.data, packed.scale, data, scale, data.numel(), scale.numel(), packed.bits, 128)
    return data, scale


@triton.jit
def _load(P, S, Z, h, t, d, T, D: tl.constexpr,
          WORDS, GROUPS, BITS: tl.constexpr,
          GROUP: tl.constexpr, CHANNEL: tl.constexpr):
    if CHANNEL:
        row, col, rows = d, t, D
    else:
        row, col, rows = t, d, T
    valid = (t < T) & (d < D)
    word = tl.load(P + (h * rows + row) * WORDS + col // (16 // BITS), valid, 0).to(tl.int32)
    code = (word >> ((16 // BITS - 1 - col % (16 // BITS)) * BITS)) & ((1 << BITS) - 1)
    param = (h * rows + row) * GROUPS + col // GROUP
    scale = tl.load(S + param, valid, 0)
    offset = tl.load(Z + param, valid, 0)
    # Launch with enable_fp_fusion=False: match torch's separate mul/sub.
    return code.to(tl.float32) * scale - offset


@triton.jit(do_not_specialize=['T', 'WORDS', 'GROUPS'])
def _decode(P, S, Z, O, T, D: tl.constexpr,
            WORDS, GROUPS, BITS: tl.constexpr,
            GROUP: tl.constexpr, CHANNEL: tl.constexpr, BLOCK: tl.constexpr):
    h = tl.program_id(1)
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = _load(P, S, Z, h, x // D, x % D, T, D, WORDS, GROUPS, BITS, GROUP, CHANNEL)
    tl.store(O + h * T * D + x, value, x < T * D)


@triton.jit
def _matmul(A, P, S, Z, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            H: tl.constexpr, HKV: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
            AB: tl.constexpr, AH: tl.constexpr, AM: tl.constexpr, AK: tl.constexpr,
            WORDS: tl.constexpr, GROUPS: tl.constexpr, BITS: tl.constexpr,
            GROUP: tl.constexpr, CHANNEL: tl.constexpr, TRANSPOSE: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    bh = tl.program_id(2)
    h = bh % H
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    r = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + r
        a = tl.load(A + (bh // H) * AB + h * AH + m[:, None] * AM + kk[None, :] * AK,
                    (m[:, None] < M) & (kk[None, :] < K), 0)
        if TRANSPOSE:
            t, d = n[None, :], kk[:, None]
        else:
            t, d = kk[:, None], n[None, :]
        b = _load(P, S, Z, h // (H // HKV), t, d, T, D, WORDS, GROUPS, BITS, GROUP, CHANNEL)
        b = b.to(A.dtype.element_ty)
        acc += tl.dot(a, b, input_precision='ieee')
    tl.store(O + bh * M * N + m[:, None] * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


def dequantize(packed, dtype):
    h, t, d = packed.shape
    out = torch.empty(packed.shape, dtype=dtype, device=packed.data.device)
    with torch.cuda.device(packed.data.device):
        _decode[(triton.cdiv(t*d, 256), h)](
            packed.data, packed.scale, packed.offset, out, t, d,
            packed.data.shape[-1], packed.scale.shape[-1], packed.bits,
            packed.group_size, packed.axis == 'channel', 256, enable_fp_fusion=False)
    return out


@triton.jit
def _gemv(A, P, S, Z, PART, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          H: tl.constexpr, HKV: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
          AB: tl.constexpr, AH: tl.constexpr, AM: tl.constexpr, AK: tl.constexpr,
          WORDS: tl.constexpr, GROUPS: tl.constexpr, BITS: tl.constexpr,
          GROUP: tl.constexpr, CHANNEL: tl.constexpr, TRANSPOSE: tl.constexpr,
          SPLITS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(1)
    bh, m = row // M, row % M
    h = bh % H
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    kk = tl.program_id(2) * BK + tl.arange(0, BK)
    a = tl.load(A + bh // H * AB + h * AH + m * AM + kk * AK, kk < K, 0).to(tl.float32)
    if TRANSPOSE:
        t, d = n[None, :], kk[:, None]
    else:
        t, d = kk[:, None], n[None, :]
    b = _load(P, S, Z, h // (H // HKV), t, d, T, D, WORDS, GROUPS, BITS, GROUP, CHANNEL)
    b = b.to(A.dtype.element_ty).to(tl.float32)
    value = tl.sum(a[:, None] * b, 0)
    tl.store(PART + (row * SPLITS + tl.program_id(2)) * N + n, value, n < N)


@triton.jit
def _gemv_reduce(PART, OUT, N: tl.constexpr, SPLITS: tl.constexpr,
                 BS: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(1)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    s = tl.arange(0, BS)
    values = tl.load(PART + (row * SPLITS + s[:, None]) * N + n[None, :],
                     (s[:, None] < SPLITS) & (n[None, :] < N), 0)
    tl.store(OUT + row * N + n, tl.sum(values, 0), n < N)


def matmul(packed, left, transpose):
    hkv, t, d = packed.shape
    a = left.unsqueeze(0) if left.ndim == 3 else left
    b, h, m, k = a.shape
    n = t if transpose else d
    out = torch.empty((b, h, m, n), dtype=a.dtype, device=a.device)
    with torch.cuda.device(a.device):
        if m <= 4:
            bk = min(triton.next_power_of_2(k), 256)
            splits = triton.cdiv(k, bk)
            partial = out if splits == 1 else torch.empty((b*h*m, splits, n), device=a.device, dtype=torch.float32)
            _gemv[(triton.cdiv(n, 32), b*h*m, splits)](
                a, packed.data, packed.scale, packed.offset, partial, m, n, k, h, hkv, t, d,
                *a.stride(), packed.data.shape[-1], packed.scale.shape[-1], packed.bits,
                packed.group_size, packed.axis == 'channel', transpose, splits, 32, bk,
                enable_fp_fusion=False, num_stages=1)
            if splits > 1:
                _gemv_reduce[(triton.cdiv(n, 32), b*h*m)](partial, out, n, splits,
                                                        triton.next_power_of_2(splits), 32)
        else:
            _matmul[(triton.cdiv(m, 16), triton.cdiv(n, 32), b*h)](
                a, packed.data, packed.scale, packed.offset, out, m, n, k, h, hkv, t, d,
                *a.stride(), packed.data.shape[-1], packed.scale.shape[-1], packed.bits,
                packed.group_size, packed.axis == 'channel', transpose, 16, 32, 32,
                enable_fp_fusion=False)
    return out.squeeze(0) if left.ndim == 3 else out


@triton.jit
def _attention_chunk(Q, K, KS, KZ, V, VS, VZ, COS, SIN, RANK,
                     ACC, MAX, SUM, M: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
                     H: tl.constexpr, HKV: tl.constexpr, PAST,
                     KW: tl.constexpr, KG: tl.constexpr, VW: tl.constexpr, VG: tl.constexpr,
                     BITS: tl.constexpr, GROUP: tl.constexpr, HALF: tl.constexpr,
                     SLOT, SPLIT_BLOCKS: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    h = tl.program_id(1)
    hk = h // (H // HKV)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    d = tl.arange(0, BD)
    q = tl.load(Q + (h * M + m[:, None]) * D + d[None, :], (m[:, None] < M) & (d[None, :] < D), 0)
    state = (SLOT + tl.program_id(2)) * H * M + h * M + m
    acc = tl.zeros((BM, BD), tl.float32)
    mx = tl.full((BM,), float('-inf'), tl.float32)
    den = tl.zeros((BM,), tl.float32)
    first = tl.program_id(2) * SPLIT_BLOCKS
    for start in range(first, tl.minimum(first + SPLIT_BLOCKS, tl.cdiv(T, BN))):
        t = start * BN + tl.arange(0, BN)
        if BITS:
            kval = _load(K, KS, KZ, hk, t[None, :], d[:, None], T, D, KW, KG, BITS, GROUP, True)
            # Match dequantize(dtype) before rotating in FP32.
            kval = kval.to(Q.dtype.element_ty).to(tl.float32)
            partner = tl.where(d < HALF, d + HALF, d - HALF)
            other = _load(K, KS, KZ, hk, t[None, :], partner[:, None], T, D, KW, KG, BITS, GROUP, True)
            other = other.to(Q.dtype.element_ty).to(tl.float32)
            c = tl.load(COS + t[None, :] * HALF + (d % HALF)[:, None],
                        (t[None, :] < T) & (d[:, None] < 2 * HALF), 1)
            s = tl.load(SIN + t[None, :] * HALF + (d % HALF)[:, None],
                        (t[None, :] < T) & (d[:, None] < 2 * HALF), 0)
            kval = tl.where(d[:, None] < 2 * HALF,
                            kval * c + tl.where(d[:, None] < HALF, -other, other) * s, kval)
            vval = _load(V, VS, VZ, hk, t[:, None], d[None, :], T, D, VW, VG, BITS, GROUP, False)
        else:
            kval = tl.load(K + (hk * T + t[None, :]) * D + d[:, None],
                           (t[None, :] < T) & (d[:, None] < D), 0)
            vval = tl.load(V + (hk * T + t[:, None]) * D + d[None, :],
                           (t[:, None] < T) & (d[None, :] < D), 0)
        scores = tl.dot(q, kval.to(Q.dtype.element_ty), input_precision='ieee') * (1.4426950408889634 / tl.sqrt(float(D)))
        rank = tl.load(RANK + t, t < T, 0x7fffffff)
        valid = (t[None, :] < T) & (rank[None, :] <= PAST + m[:, None])
        scores = tl.where(valid, scores, float('-inf'))
        new_mx = tl.maximum(mx, tl.max(scores, 1))
        safe_mx = tl.where(new_mx == float('-inf'), 0., new_mx)
        alpha = tl.exp2(mx - safe_mx)
        prob = tl.exp2(scores - safe_mx[:, None])
        acc = acc * alpha[:, None] + tl.dot(prob.to(Q.dtype.element_ty), vval.to(Q.dtype.element_ty), input_precision='ieee')
        den = den * alpha + tl.sum(prob, 1)
        mx = new_mx
    tl.store(ACC + state[:, None] * D + d[None, :], acc, (m[:, None] < M) & (d[None, :] < D))
    tl.store(MAX + state, mx, m < M)
    tl.store(SUM + state, den, m < M)


@triton.jit
def _merge_attention(ACC, MAX, SUM, OUT, ROWS: tl.constexpr, D: tl.constexpr,
                     SLOTS: tl.constexpr, BS: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    s = tl.arange(0, BS)
    d = tl.arange(0, BD)
    mx = tl.load(MAX + s * ROWS + row, s < SLOTS, float('-inf'))
    den = tl.load(SUM + s * ROWS + row, s < SLOTS, 0)
    weights = tl.exp2(mx - tl.max(mx, 0))
    values = tl.load(ACC + (s[:, None] * ROWS + row) * D + d[None, :],
                     (s[:, None] < SLOTS) & (d[None, :] < D), 0)
    result = tl.sum(values * weights[:, None], 0) / tl.sum(den * weights, 0)
    tl.store(OUT + row * D + d, result, d < D)


def attention_chunks(query, chunks, past_length):
    """Split-K online softmax; merge partials without dense KV or QK allocation."""
    _, h, m, d = query.shape
    query = query.contiguous()
    splits = [triton.cdiv(chunk.key.shape[1], 256) for chunk in chunks]
    slots = sum(splits)
    acc = torch.empty((slots, h, m, d), device=query.device, dtype=torch.float32)
    mx = torch.empty((slots, h, m), device=query.device, dtype=torch.float32)
    den = torch.empty_like(mx)
    out = torch.empty_like(query)
    with torch.cuda.device(query.device):
        slot = 0
        for i, chunk in enumerate(chunks):
            k, v = chunk.key, chunk.value
            packed = hasattr(k, 'bits')
            hk, t, _ = k.shape
            _attention_chunk[(triton.cdiv(m, 16), h, splits[i])](
                query, k.data if packed else k, k.scale if packed else k, k.offset if packed else k,
                v.data if packed else v, v.scale if packed else v, v.offset if packed else v,
                chunk.cos, chunk.sin, chunk.ranks, acc, mx, den, m, t, d, h, hk, past_length,
                k.data.shape[-1] if packed else 0, k.scale.shape[-1] if packed else 0,
                v.data.shape[-1] if packed else 0, v.scale.shape[-1] if packed else 0,
                k.bits if packed else 0, k.group_size if packed else 0,
                chunk.cos.shape[-1], slot, 8, 16, 32, triton.next_power_of_2(d),
                enable_fp_fusion=False, num_stages=1)
            slot += splits[i]
        _merge_attention[(h*m,)](acc, mx, den, out, h*m, d, slots,
                                 triton.next_power_of_2(slots), triton.next_power_of_2(d))
    return out
