"""Batched ragged restoration and GQA attention for the dense inference backend."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _read(P, S, Z, token, channel, N, D: tl.constexpr, BITS, GROUP, WORDS, GROUPS,
          TS, DS, CHANNEL: tl.constexpr, DTYPE: tl.constexpr):
    valid = (token < N) & (channel < D)
    if BITS == 0:
        value = tl.load(P.to(tl.pointer_type(DTYPE)) + token*TS + channel*DS, valid, 0).to(tl.float32)
    else:
        row, col = (channel, token) if CHANNEL else (token, channel)
        pack = 16 // BITS
        word = tl.load(P.to(tl.pointer_type(tl.int16)) + row*WORDS + col//pack, valid, 0).to(tl.int32)
        code = (word >> ((pack-1-col%pack)*BITS)) & ((1 << BITS)-1)
        param = row*GROUPS + col//GROUP
        scale = tl.load(S.to(tl.pointer_type(tl.float32)) + param, valid, 0)
        offset = tl.load(Z.to(tl.pointer_type(tl.float32)) + param, valid, 0)
        value = libdevice.sub_rn(libdevice.mul_rn(code.to(tl.float32), scale), offset).to(DTYPE).to(tl.float32)
    return value


@triton.jit
def _restore(DESC, KOUT, VOUT, FREQ, D: tl.constexpr, HALF: tl.constexpr,
             DISTANCE, DTYPE: tl.constexpr, RANK_STEPS: tl.constexpr,
             BT: tl.constexpr, BD: tl.constexpr):
    row = DESC + tl.program_id(1)*23
    kp, ks, kz = tl.load(row), tl.load(row+1), tl.load(row+2)
    vp, vs, vz = tl.load(row+3), tl.load(row+4), tl.load(row+5)
    ip, other, other_n, n = tl.load(row+6), tl.load(row+7), tl.load(row+8), tl.load(row+9)
    destination, logical, source = tl.load(row+10), tl.load(row+11), tl.load(row+12)
    bits, group = tl.load(row+13), tl.load(row+14)
    kw, kg, vw, vg = tl.load(row+15), tl.load(row+16), tl.load(row+17), tl.load(row+18)
    kt, kd, vt, vd = tl.load(row+19), tl.load(row+20), tl.load(row+21), tl.load(row+22)
    t = tl.program_id(0)*BT + tl.arange(0, BT)
    d = tl.arange(0, BD)
    if ip == 0:
        ids = t
    else:
        ids = tl.load(ip.to(tl.pointer_type(tl.int32))+t, t<n, 0)
    # Each partition is sorted by original token ID. Merge rank is its own
    # index plus lower_bound in the other partition; no frame/head argsort.
    rank = t
    if other_n > 0:
        lo = tl.full((BT,), 0, tl.int32)
        hi = tl.full((BT,), 0, tl.int32) + other_n.to(tl.int32)
        for _ in range(RANK_STEPS):
            mid = (lo+hi)//2
            other_id = tl.load(other.to(tl.pointer_type(tl.int32))+mid, (mid<other_n) & (t<n), 0x7fffffff)
            go_right = (lo<hi) & (other_id<ids)
            hi = tl.where((lo<hi) & ~go_right, mid, hi)
            lo = tl.where(go_right, mid+1, lo)
        rank += lo
    kval = _read(kp, ks, kz, t[:, None], d[None, :], n, D, bits, group, kw, kg, kt, kd, True, DTYPE)
    partner = tl.where(d<2*HALF, tl.where(d<HALF, d+HALF, d-HALF), d)
    other_k = _read(kp, ks, kz, t[:, None], partner[None, :], n, D, bits, group, kw, kg, kt, kd, True, DTYPE)
    freq = tl.load(FREQ+d%HALF, d<2*HALF, 0).to(tl.float32)
    angle = (logical+rank-ids-source).to(tl.float32)[:, None]*DISTANCE*freq[None, :]
    rotated = kval*libdevice.cos(angle) + tl.where(d[None, :]<HALF, -other_k, other_k)*libdevice.sin(angle)
    kval = tl.where(d[None, :]<2*HALF, rotated, kval)
    vval = _read(vp, vs, vz, t[:, None], d[None, :], n, D, bits, group, vw, vg, vt, vd, False, DTYPE)
    out = (destination+rank[:, None])*D+d[None, :]
    mask = (t[:, None]<n) & (d[None, :]<D)
    tl.store(KOUT+out, kval, mask)
    tl.store(VOUT+out, vval, mask)


@triton.jit(do_not_specialize=['M', 'GENERATED'])
def _append(Q, K, V, QOUT, KOUT, VOUT, OFFSETS, LENGTHS, FREQ,
            M, GENERATED, D: tl.constexpr, G: tl.constexpr, HALF: tl.constexpr,
            DISTANCE, BT: tl.constexpr, BD: tl.constexpr):
    h = tl.program_id(1)
    hk = h//G
    t = tl.program_id(0)*BT+tl.arange(0, BT)
    d = tl.arange(0, BD)
    partner = tl.where(d<2*HALF, tl.where(d<HALF, d+HALF, d-HALF), d)
    base = tl.load(LENGTHS+hk)+GENERATED
    angle = (base+t).to(tl.float32)[:, None]*DISTANCE*tl.load(FREQ+d%HALF, d<2*HALF, 0)[None, :]
    cos, sin = libdevice.cos(angle), libdevice.sin(angle)
    mask = (t[:, None]<M) & (d[None, :]<D)
    q = tl.load(Q+(h*M+t[:, None])*D+d[None, :], mask, 0).to(tl.float32)
    qp = tl.load(Q+(h*M+t[:, None])*D+partner[None, :], mask, 0).to(tl.float32)
    qr = q*cos + tl.where(d[None, :]<HALF, -qp, qp)*sin
    tl.store(QOUT+(h*M+t[:, None])*D+d[None, :], tl.where(d[None, :]<2*HALF, qr, q), mask)
    # All query heads rotate together; only the first in each GQA group writes KV.
    if h%G == 0:
        k = tl.load(K+(hk*M+t[:, None])*D+d[None, :], mask, 0).to(tl.float32)
        kp = tl.load(K+(hk*M+t[:, None])*D+partner[None, :], mask, 0).to(tl.float32)
        kr = k*cos + tl.where(d[None, :]<HALF, -kp, kp)*sin
        v = tl.load(V+(hk*M+t[:, None])*D+d[None, :], mask, 0)
        out = (tl.load(OFFSETS+hk)+base+t[:, None])*D+d[None, :]
        tl.store(KOUT+out, tl.where(d[None, :]<2*HALF, kr, k), mask)
        tl.store(VOUT+out, v, mask)


@triton.jit(do_not_specialize=['M', 'GENERATED'])
def _attend(Q, K, V, OFFSETS, LENGTHS, ACC, MAX, SUM,
            M, GENERATED, H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr, BLOCKS: tl.constexpr):
    h, split = tl.program_id(1), tl.program_id(2)
    hk = h//G
    m = tl.program_id(0)*BM+tl.arange(0, BM)
    d = tl.arange(0, BD)
    past = tl.load(LENGTHS+hk)+GENERATED
    total = past+M
    offset = tl.load(OFFSETS+hk)
    q = tl.load(Q+(h*M+m[:, None])*D+d[None, :], (m[:, None]<M) & (d[None, :]<D), 0)
    acc = tl.zeros((BM, BD), tl.float32)
    mx = tl.full((BM,), float('-inf'), tl.float32)
    den = tl.zeros((BM,), tl.float32)
    first = split*BLOCKS
    for block in range(first, tl.minimum(first+BLOCKS, tl.cdiv(total, BN))):
        t = block*BN+tl.arange(0, BN)
        k = tl.load(K+(offset+t[None, :])*D+d[:, None], (t[None, :]<total) & (d[:, None]<D), 0)
        v = tl.load(V+(offset+t[:, None])*D+d[None, :], (t[:, None]<total) & (d[None, :]<D), 0)
        scores = tl.dot(q, k, input_precision='ieee')*(1.4426950408889634 / tl.sqrt(float(D)))
        scores = tl.where((t[None, :]<total) & (t[None, :]<=past+m[:, None]), scores, float('-inf'))
        new_mx = tl.maximum(mx, tl.max(scores, 1))
        safe_mx = tl.where(new_mx == float('-inf'), 0., new_mx)
        alpha = tl.exp2(mx-safe_mx)
        prob = tl.exp2(scores-safe_mx[:, None])
        acc = acc*alpha[:, None] + tl.dot(prob.to(q.dtype), v, input_precision='ieee')
        den = den*alpha+tl.sum(prob, 1)
        mx = new_mx
    state = (split*H+h)*M+m
    tl.store(ACC+state[:, None]*D+d[None, :], acc, (m[:, None]<M) & (d[None, :]<D))
    tl.store(MAX+state, mx, m<M)
    tl.store(SUM+state, den, m<M)


@triton.jit(do_not_specialize=['ROWS', 'SPLITS'])
def _merge(ACC, MAX, SUM, OUT, ROWS, SPLITS, D: tl.constexpr, BS: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    s, d = tl.arange(0, BS), tl.arange(0, BD)
    mx = tl.load(MAX+s*ROWS+row, s<SPLITS, float('-inf'))
    den = tl.load(SUM+s*ROWS+row, s<SPLITS, 0)
    weight = tl.exp2(mx-tl.max(mx, 0))
    acc = tl.load(ACC+(s[:, None]*ROWS+row)*D+d[None, :], (s[:, None]<SPLITS) & (d[None, :]<D), 0)
    out = tl.sum(acc*weight[:, None], 0)/tl.sum(den*weight, 0)
    tl.store(OUT+row*D+d, out, d<D)


def materialize(cache, descriptors, max_tokens, rope, source_dtype):
    desc = torch.tensor(descriptors, dtype=torch.int64, device=cache.key.device)
    dim = cache.key.shape[-1]
    dtype = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}[source_dtype]
    with torch.cuda.device(cache.key.device):
        _restore[(triton.cdiv(max_tokens, 16), len(descriptors))](
            desc, cache.key, cache.value, rope.inv_freq, dim, rope.inv_freq.numel(), rope.distance_scale,
            dtype, max_tokens.bit_length(), 16, triton.next_power_of_2(dim), enable_fp_fusion=False)


def append_and_attend(cache, query, key, value, rope):
    _, h, m, d = query.shape
    group = h//len(cache.host_lengths)
    splits = triton.cdiv(cache.max_length+m, 512)
    shape = (splits, h, m, d)
    if m != 1 or cache.scratch is None or cache.scratch[0].shape != shape:
        acc = torch.empty(shape, dtype=torch.float32, device=query.device)
        mx = torch.empty(shape[:-1], dtype=torch.float32, device=query.device)
        den = torch.empty_like(mx)
        # Retain only the small single-token workspace. Keeping each layer's
        # prefill partials until decoding would multiply the peak by layer count.
        cache.scratch = (acc, mx, den) if m == 1 else None
    else:
        acc, mx, den = cache.scratch
    rotated, out = torch.empty_like(query), torch.empty_like(query)
    with torch.cuda.device(query.device):
        _append[(triton.cdiv(m, 16), h)](query, key, value, rotated, cache.key, cache.value,
            cache.offsets, cache.base_lengths, rope.inv_freq, m, cache.generated, d, group,
            rope.inv_freq.numel(), rope.distance_scale, 16, max(16, triton.next_power_of_2(d)), enable_fp_fusion=False)
        _attend[(triton.cdiv(m, 16), h, splits)](rotated, cache.key, cache.value,
            cache.offsets, cache.base_lengths, acc, mx, den, m, cache.generated, h, group, d,
            16, 64, max(16, triton.next_power_of_2(d)), 8, enable_fp_fusion=False, num_stages=1)
        _merge[(h*m,)](acc, mx, den, out, h*m, splits, d, triton.next_power_of_2(splits), max(16, triton.next_power_of_2(d)))
    return out
