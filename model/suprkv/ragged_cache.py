"""Question-local flat KV with independent head lengths and reusable decode space.

Persistent frames remain immutable. Only this single-question inference cache
is mutable; a retrieval probe writes scratch tail slots without committing them.
"""
from dataclasses import dataclass, field
import importlib.util

import torch

from .quantization import PackedKV, CompactPackedKV


def supported(query):
    return (query.is_cuda and torch.version.cuda is not None and query.shape[0] == 1
            and query.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and 0 < query.shape[-1] <= 256 and not torch.is_grad_enabled()
            and importlib.util.find_spec('triton') is not None)


@dataclass
class RaggedKVCache:
    key: torch.Tensor
    value: torch.Tensor
    offsets: torch.Tensor
    base_lengths: torch.Tensor
    host_offsets: tuple
    host_lengths: tuple
    generated: int = 0
    scratch: object = field(default=None, repr=False)

    @classmethod
    def allocate(cls, lengths, dim, dtype, device, reserve=1024):
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length + reserve)
        key = torch.empty((offsets[-1], dim), dtype=dtype, device=device)
        return cls(key, torch.empty_like(key),
                   torch.tensor(offsets[:-1], dtype=torch.int64, device=device),
                   torch.tensor(lengths, dtype=torch.int32, device=device),
                   tuple(offsets), tuple(lengths))

    @property
    def max_length(self):
        return max(self.host_lengths) + self.generated

    def ensure_capacity(self, count):
        if all(n + self.generated + count <= end - start for n, start, end in
               zip(self.host_lengths, self.host_offsets, self.host_offsets[1:])):
            return
        # An unusually long prompt can exceed the reserved tail. Grow once for
        # that prompt, not by copying the whole history at every decode token.
        reserve = max(1024, 2 * (self.generated + count))
        grown = self.allocate(self.host_lengths, self.key.shape[-1], self.key.dtype, self.key.device, reserve)
        for old, new, n in zip(self.host_offsets, grown.host_offsets, self.host_lengths):
            length = n + self.generated
            grown.key[new:new+length].copy_(self.key[old:old+length])
            grown.value[new:new+length].copy_(self.value[old:old+length])
        self.key, self.value = grown.key, grown.value
        self.offsets, self.host_offsets = grown.offsets, grown.host_offsets

    def attend(self, query, key, value, rope, *, commit):
        if not supported(query) or query.shape[-2] < 1 or key.shape != value.shape or key.shape[:3] != (
                1, len(self.host_lengths), query.shape[-2]) or any(
                x.device != self.key.device or x.dtype != self.key.dtype or not x.is_contiguous()
                for x in (query, key, value)) or not (
                self.key.shape[-1] == query.shape[-1] == key.shape[-1]):
            raise ValueError('Ragged attention requires same-device inference GQA tensors')
        if query.shape[1] % len(self.host_lengths):
            raise ValueError('Invalid ragged GQA group count')
        self.ensure_capacity(query.shape[-2])
        from .ragged_kernels import append_and_attend
        result = append_and_attend(self, query, key, value, rope)
        if commit:
            self.generated += query.shape[-2]
        return result

    def as_headwise(self):
        """Compatibility fallback for the existing sliding/sink window path."""
        from .attention import RotaryCache
        from .packed_attention import HeadwiseRotaryCache
        heads = []
        for start, n in zip(self.host_offsets, self.host_lengths):
            length = n + self.generated
            heads.append(RotaryCache(self.key[start:start+length][None, None],
                                     self.value[start:start+length][None, None],
                                     torch.arange(length, device=self.key.device)))
        return HeadwiseRotaryCache(tuple(heads))


def _range(entry, high, head):
    indices = entry.high_indices if high else entry.low_indices
    offsets = entry.high_offsets if high else entry.low_offsets
    if offsets is None:
        return 0, indices.numel()
    return offsets[head], offsets[head+1] - offsets[head]


def _payload(payload, head, start, compact):
    """Raw device addresses and strides; no per-head tensor views or launches."""
    if isinstance(payload, (PackedKV, CompactPackedKV)):
        if isinstance(payload, CompactPackedKV):
            data_start = payload.data_offsets[head]
            param_start = payload.parameter_offsets[head]
        else:
            data_start = head * payload.data.stride(0)
            param_start = head * payload.scale.stride(0)
        return (payload.data.data_ptr() + 2 * data_start,
                payload.scale.data_ptr() + 4 * param_start,
                payload.offset.data_ptr() + 4 * param_start, 0, 0)
    base = start * payload.stride(0) if compact else head * payload.stride(0)
    return (payload.data_ptr() + base * payload.element_size(), 0, 0,
            payload.stride(-2), payload.stride(-1))


def from_frames(sink, frames, layer, rope, dtype, reserve=1024):
    """One batched kernel decodes, orders and rebases all selected head regions."""
    sk, sv = sink
    heads, sink_length, dim = sk.shape
    lengths = [sink_length] * heads
    for frame in frames:
        entry = frame.layers[layer]
        for head in range(heads):
            lengths[head] += _range(entry, True, head)[1] + _range(entry, False, head)[1]
    result = RaggedKVCache.allocate(lengths, dim, dtype, sk.device, reserve)
    descriptors, max_tokens = [], 1
    # Fields: K/KS/KZ/V/VS/VZ, indices/other indices/other count/count,
    # output offset/logical start/source RoPE start/bits/group,
    # K words/groups/V words/groups, dense K and V token/channel strides.
    for head in range(heads):
        logical = 0
        if sink_length:
            kp, ks, kz, kt, kd = _payload(sk, head, 0, False)
            vp, vs, vz, vt, vd = _payload(sv, head, 0, False)
            descriptors.append([kp, ks, kz, vp, vs, vz, 0, 0, 0, sink_length,
                result.host_offsets[head], 0, 0, 0, 1, 0, 0, 0, 0, kt, kd, vt, vd])
            max_tokens = max(max_tokens, sink_length)
            logical += sink_length
        for frame in frames:
            entry = frame.layers[layer]
            ranges = [_range(entry, True, head), _range(entry, False, head)]
            for part, high in enumerate((True, False)):
                start, n = ranges[part]
                if not n:
                    continue
                other_start, other_n = ranges[1-part]
                k, v, ids, other_ids, offsets = (
                    (entry.high_k, entry.high_v, entry.high_indices, entry.low_indices, entry.high_offsets)
                    if high else (entry.low_k, entry.low_v, entry.low_indices, entry.high_indices, entry.low_offsets))
                packed = isinstance(k, (PackedKV, CompactPackedKV))
                bits, group = (k.bits, k.group_size) if packed else (0, 1)
                kp, ks, kz, kt, kd = _payload(k, head, start, offsets is not None)
                vp, vs, vz, vt, vd = _payload(v, head, start, offsets is not None)
                descriptors.append([kp, ks, kz, vp, vs, vz,
                    ids.data_ptr()+start*ids.element_size(),
                    other_ids.data_ptr()+other_start*other_ids.element_size() if other_n else 0,
                    other_n, n, result.host_offsets[head]+logical, logical, frame.rope_start, bits, group,
                    ((n+7)//8*8)//(16//bits) if packed else 0, (n+group-1)//group,
                    ((dim+7)//8*8)//(16//bits) if packed else 0, (dim+group-1)//group,
                    kt, kd, vt, vd])
                max_tokens = max(max_tokens, n, other_n)
            logical += ranges[0][1] + ranges[1][1]
    if descriptors:
        from .ragged_kernels import materialize
        materialize(result, descriptors, max_tokens, rope, sk.dtype)
    return result
