"""Batch a budget event without sharing storage between canonical payloads."""
from dataclasses import replace
import importlib.util

import torch

from .quantization import PackedKV, CompactPackedKV


def supported(entries, group_size):
    def payload_ok(payload):
        if isinstance(payload, (PackedKV, CompactPackedKV)):
            return all(x.is_cuda and x.is_contiguous() for x in (payload.data, payload.scale, payload.offset))
        return payload.is_cuda and payload.dtype in (torch.float16, torch.bfloat16, torch.float32)
    return (bool(entries) and group_size <= 4096 and torch.version.cuda is not None
            and importlib.util.find_spec('triton') is not None
            and all(e.high_indices.is_cuda and e.dtype in (torch.float16, torch.bfloat16, torch.float32)
                    and payload_ok(e.high_k) and payload_ok(e.high_v)
                    for e in entries))


def lower(entries, group_size):
    """One precision step for every eligible entry; publish only after success."""
    from .batch_kernels import quantize_many, shrink_many
    fresh, packed, owners = [], [], []
    for index, entry in enumerate(entries):
        if entry.bits == 2 or not entry.high_indices.numel():
            continue
        for name, axis in (('high_k', 'channel'), ('high_v', 'token')):
            payload = getattr(entry, name)
            if entry.bits == 16:
                fresh.append((payload, entry.high_offsets, group_size, axis))
                owners.append((index, name, True, len(fresh)-1))
            else:
                packed.append(payload)
                owners.append((index, name, False, len(packed)-1))
    quantized = quantize_many(fresh)
    lowered = shrink_many(packed)
    changes = {}
    for index, name, initial, position in owners:
        result = quantized[position] if initial else lowered[position]
        changes.setdefault(index, {})[name] = result
    return tuple(replace(entry, **changes[i], bits=changes[i]['high_k'].bits) if i in changes else entry
                 for i, entry in enumerate(entries))


def allocate_quantized(value, offsets, group, axis):
    """Return the original layout plus per-head generic row/column descriptors."""
    compact = offsets is not None
    if compact:
        if (value.ndim != 2 or offsets[0] != 0 or offsets[-1] != len(value)
                or any(b <= a for a, b in zip(offsets, offsets[1:]))):
            raise ValueError('Retained high regions must be nonempty in every head')
        lengths = [b-a for a, b in zip(offsets, offsets[1:])]
    else:
        if value.ndim != 3 or min(value.shape) < 1:
            raise ValueError('Expected nonempty [heads, tokens, channels] KV')
        lengths = [value.shape[1]]*value.shape[0]
    dim = value.shape[-1]
    data_offsets, parameter_offsets = [0], [0]
    for n in lengths:
        rows, length = (dim, n) if axis == 'channel' else (n, dim)
        data_offsets.append(data_offsets[-1]+rows*((length+7)//8*8)//2)
        parameter_offsets.append(parameter_offsets[-1]+rows*((length+group-1)//group))
    data = torch.empty(data_offsets[-1], dtype=torch.int16, device=value.device)
    scale = torch.empty(parameter_offsets[-1], dtype=torch.float32, device=value.device)
    offset = torch.empty_like(scale)
    if compact:
        result = CompactPackedKV(data, scale, offset, tuple(offsets), dim, 8, group, axis,
                                 tuple(data_offsets), tuple(parameter_offsets))
    else:
        rows, length = (dim, lengths[0]) if axis == 'channel' else (lengths[0], dim)
        result = PackedKV(data.view(len(lengths), rows, (length+7)//8*4),
                          scale.view(len(lengths), rows, (length+group-1)//group),
                          offset.view(len(lengths), rows, (length+group-1)//group),
                          tuple(value.shape), 8, group, axis)
    descriptors = []
    for h, n in enumerate(lengths):
        start = offsets[h]*value.stride(0) if compact else h*value.stride(0)
        rows, length = (dim, n) if axis == 'channel' else (n, dim)
        sr, sc = (value.stride(-1), value.stride(-2)) if axis == 'channel' else value.stride()[-2:]
        descriptors.append((value.data_ptr()+start*value.element_size(),
            data.data_ptr()+2*data_offsets[h], scale.data_ptr()+4*parameter_offsets[h],
            offset.data_ptr()+4*parameter_offsets[h], rows, length, sr, sc))
    return result, descriptors
