"""Restore invalidated frame/layer payloads in batches, without RoPE rebasing."""
from collections import defaultdict
import importlib.util

import torch

from .quantization import PackedKV, CompactPackedKV
from .ragged_cache import _range, _payload


def supported(entry):
    return (entry.high_indices.is_cuda and torch.version.cuda is not None
            and entry.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and importlib.util.find_spec('triton') is not None)


def materialize_many(entries):
    groups, results = defaultdict(list), []
    for entry in entries:
        device, dtype, dim = entry.high_indices.device, entry.dtype, entry.representative.shape[-1]
        compact = entry.high_offsets is not None
        parts = []
        if not compact:
            n = entry.high_indices.numel()+entry.low_indices.numel()
            k = torch.empty((entry.heads, n, dim), dtype=dtype, device=device)
            v, ids = torch.empty_like(k), torch.empty(n, dtype=torch.int32, device=device)
            parts.append((k, v, ids))
        for head in range(entry.heads):
            ranges = [_range(entry, True, head), _range(entry, False, head)]
            n = sum(count for _, count in ranges)
            if compact:
                k = torch.empty((1, n, dim), dtype=dtype, device=device)
                v, ids = torch.empty_like(k), torch.empty(n, dtype=torch.int32, device=device)
                parts.append((k, v, ids))
            shift = 0 if compact else head*n*dim*dtype.itemsize
            for part, high in enumerate((True, False)):
                start, count = ranges[part]
                if not count:
                    continue
                other_start, other_n = ranges[1-part]
                key, value, index, other, offsets = (
                    (entry.high_k, entry.high_v, entry.high_indices, entry.low_indices, entry.high_offsets)
                    if high else (entry.low_k, entry.low_v, entry.low_indices, entry.high_indices, entry.low_offsets))
                packed = isinstance(key, (PackedKV, CompactPackedKV))
                bits, group = (key.bits, key.group_size) if packed else (0, 1)
                kp, ks, kz, kt, kd = _payload(key, head, start, offsets is not None)
                vp, vs, vz, vt, vd = _payload(value, head, start, offsets is not None)
                groups[(device, dtype, dim)].append((kp, ks, kz, vp, vs, vz,
                    index.data_ptr()+start*index.element_size(),
                    other.data_ptr()+other_start*other.element_size() if other_n else 0,
                    other_n, count, bits, group,
                    ((count+7)//8*8)//(16//bits) if packed else 0, (count+group-1)//group,
                    ((dim+7)//8*8)//(16//bits) if packed else 0, (dim+group-1)//group,
                    kt, kd, vt, vd, k.data_ptr()+shift, v.data_ptr()+shift,
                    ids.data_ptr() if compact or head == 0 else 0))
        results.append(tuple(parts))
    from .batch_kernels import restore_many
    for (device, dtype, dim), descriptors in groups.items():
        restore_many(descriptors, device, dtype, dim)
    return results
