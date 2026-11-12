"""Multi-payload CUDA launches; each result retains independent storage."""
from dataclasses import replace
from collections import defaultdict

import torch
import triton
import triton.language as tl

from .packed_kernels import _encode_group, _lower_codes
from .ragged_kernels import _read


@triton.jit
def _restore_batch(DESC, D: tl.constexpr, DTYPE: tl.constexpr, STEPS: tl.constexpr,
                   BT: tl.constexpr, BD: tl.constexpr):
    row = DESC+tl.program_id(1)*23
    kp, ks, kz = tl.load(row), tl.load(row+1), tl.load(row+2)
    vp, vs, vz = tl.load(row+3), tl.load(row+4), tl.load(row+5)
    ip, other = tl.load(row+6), tl.load(row+7)
    other_n, n = tl.load(row+8), tl.load(row+9)
    bits, group = tl.load(row+10), tl.load(row+11)
    kw, kg, vw, vg = tl.load(row+12), tl.load(row+13), tl.load(row+14), tl.load(row+15)
    kt, kd, vt, vd = tl.load(row+16), tl.load(row+17), tl.load(row+18), tl.load(row+19)
    kout = tl.load(row+20).to(tl.pointer_type(DTYPE))
    vout = tl.load(row+21).to(tl.pointer_type(DTYPE))
    iout = tl.load(row+22)
    t = tl.program_id(0)*BT+tl.arange(0, BT)
    d = tl.arange(0, BD)
    ids = tl.load(ip.to(tl.pointer_type(tl.int32))+t, t<n, 0)
    rank = t
    if other_n>0:
        lo = tl.full((BT,), 0, tl.int32)
        hi = lo+other_n.to(tl.int32)
        for _ in range(STEPS):
            mid = (lo+hi)//2
            other_id = tl.load(other.to(tl.pointer_type(tl.int32))+mid, (mid<other_n) & (t<n), 0x7fffffff)
            right = (lo<hi) & (other_id<ids)
            hi = tl.where((lo<hi) & ~right, mid, hi)
            lo = tl.where(right, mid+1, lo)
        rank += lo
    k = _read(kp, ks, kz, t[:, None], d[None, :], n, D, bits, group, kw, kg, kt, kd, True, DTYPE)
    v = _read(vp, vs, vz, t[:, None], d[None, :], n, D, bits, group, vw, vg, vt, vd, False, DTYPE)
    mask = (t[:, None]<n) & (d[None, :]<D)
    tl.store(kout+rank[:, None]*D+d[None, :], k, mask)
    tl.store(vout+rank[:, None]*D+d[None, :], v, mask)
    if iout!=0:
        tl.store(iout.to(tl.pointer_type(tl.int32))+rank, ids, t<n)


def restore_many(descriptors, device, dtype, dim):
    desc = torch.tensor(descriptors, dtype=torch.int64, device=device)
    maximum = max(max(row[8:10]) for row in descriptors)
    scalar = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}[dtype]
    with torch.cuda.device(device):
        _restore_batch[(triton.cdiv(maximum, 16), len(descriptors))](desc, dim, scalar,
            maximum.bit_length(), 16, max(16, triton.next_power_of_2(dim)), enable_fp_fusion=False)


@triton.jit
def _quantize_batch(DESC, ERROR, DTYPE: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr):
    job, task = tl.program_id(0), tl.program_id(1)
    desc = DESC + task*8
    xp = tl.load(desc).to(tl.pointer_type(DTYPE))
    p = tl.load(desc+1).to(tl.pointer_type(tl.int16))
    s = tl.load(desc+2).to(tl.pointer_type(tl.float32))
    z = tl.load(desc+3).to(tl.pointer_type(tl.float32))
    rows, length = tl.load(desc+4), tl.load(desc+5)
    sr, sc = tl.load(desc+6), tl.load(desc+7)
    groups = tl.cdiv(length, GROUP)
    if job < rows*groups:
        row, g = job//groups, job%groups
        i = tl.arange(0, BLOCK)
        col = g*GROUP+i
        valid = (i<GROUP) & (col<length)
        x = tl.load(xp+row*sr+col*sc, valid, 0).to(tl.float32)
        if tl.sum(((x!=x) | (tl.abs(x)==float('inf'))) & valid, 0)>0:
            tl.atomic_or(ERROR, 1)
        word, scale, offset = _encode_group(x, valid, 8, BLOCK)
        words = tl.cdiv(length, 8)*4
        wi = tl.arange(0, BLOCK//2)
        wc = g*(GROUP//2)+wi
        tl.store(p+row*words+wc, word, (wi<GROUP//2) & (wc<words))
        tl.store(s+row*groups+g, scale)
        tl.store(z+row*groups+g, offset)


@triton.jit
def _shrink_batch(DESC, BITS: tl.constexpr, BLOCK: tl.constexpr):
    desc = DESC+tl.program_id(1)*6
    p = tl.load(desc).to(tl.pointer_type(tl.int16))
    s = tl.load(desc+1).to(tl.pointer_type(tl.float32))
    out = tl.load(desc+2).to(tl.pointer_type(tl.int16))
    os = tl.load(desc+3).to(tl.pointer_type(tl.float32))
    words, params = tl.load(desc+4), tl.load(desc+5)
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    N: tl.constexpr = BITS//2
    lanes = tl.arange(0, 16//N)
    indices = i[:, None]*(16//N)+lanes[None, :]
    word = tl.load(p+indices//(16//BITS), i[:, None]<words, 0).to(tl.int32)
    code = (word >> ((16//BITS-1-indices%(16//BITS))*BITS)) & ((1<<BITS)-1)
    code = _lower_codes(code, N)
    result = tl.sum(code << ((16//N-1-lanes[None, :])*N), 1)
    tl.store(out+i, result, i<words)
    scale = tl.load(s+i, i<params, 0)
    tl.store(os+i, scale*(2**N+1), i<params)


def quantize_many(requests):
    from .batch_quantization import allocate_quantized
    results, groups = [], defaultdict(list)
    for value, offsets, group, axis in requests:
        result, descriptors = allocate_quantized(value, offsets, group, axis)
        results.append(result)
        groups[(value.device, value.dtype, group)].extend(descriptors)
    errors = {}
    for (device, dtype, group), descriptors in groups.items():
        if device not in errors:
            errors[device] = torch.zeros((), dtype=torch.int32, device=device)
        desc = torch.tensor(descriptors, dtype=torch.int64, device=device)
        jobs = max(row[4]*triton.cdiv(row[5], group) for row in descriptors)
        scalar = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}[dtype]
        with torch.cuda.device(device):
            _quantize_batch[(jobs, len(descriptors))](desc, errors[device], scalar, group,
                                                      triton.next_power_of_2(group), enable_fp_fusion=False)
    # One readback per device for the entire event, instead of every K and V.
    if any(error.item() for error in errors.values()):
        raise ValueError('Expected nonempty finite [heads, tokens, channels] KV')
    return results


def shrink_many(payloads):
    from .quantization import CompactPackedKV
    results, groups = [], defaultdict(list)
    for p in payloads:
        data = torch.empty((*p.data.shape[:-1], p.data.shape[-1]//2), dtype=p.data.dtype, device=p.data.device)
        scale = torch.empty_like(p.scale)
        changes = dict(data=data, scale=scale, bits=p.bits//2)
        if isinstance(p, CompactPackedKV):
            changes['data_offsets'] = tuple(x//2 for x in p.data_offsets)
        results.append(replace(p, **changes))
        groups[(p.data.device, p.bits)].append((p.data.data_ptr(), p.scale.data_ptr(), data.data_ptr(),
                                              scale.data_ptr(), data.numel(), scale.numel()))
    for (device, bits), descriptors in groups.items():
        desc = torch.tensor(descriptors, dtype=torch.int64, device=device)
        count = max(max(row[4:]) for row in descriptors)
        with torch.cuda.device(device):
            _shrink_batch[(triton.cdiv(count, 128), len(descriptors))](desc, bits, 128)
    return results
