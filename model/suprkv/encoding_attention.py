"""Batched variable-length attention for already-rotated video encoding KV."""
import torch
from dataclasses import dataclass


@dataclass
class EncodingSegments:
    """Ordered post-RoPE pieces; join only at the consuming attention layer."""
    parts: tuple

    def materialize(self):
        from .attention import EncodingKV
        return EncodingKV(torch.cat([k for k, _ in self.parts], 1)[None],
                          torch.cat([v for _, v in self.parts], 1)[None])


def supported(query):
    return (query.is_cuda and torch.version.cuda is not None and query.shape[0] == 1
            and query.shape[-2] > 0 and query.shape[-1] in (16, 32, 64, 128)
            and query.dtype in (torch.float16, torch.bfloat16)
            and not torch.is_grad_enabled())


def attend(query, key, value, heads):
    """Each GQA group sees its own prefix and the causal current-frame tail.

    Inputs are post-RoPE. No positions are changed, and no persistent encoding
    buffers are retained. CPU/unsupported calls use the caller's legacy path.
    """
    if not supported(query) or not heads or query.shape[1] % len(heads):
        raise ValueError('Batched encoding requires CUDA inference GQA tensors')
    count, dim = query.shape[-2:]
    if key.shape != value.shape or key.shape != (1, len(heads), count, dim):
        raise ValueError('Current-frame K/V must match the query and KV head count')
    if any(t.device != query.device or t.dtype != query.dtype for t in (key, value)):
        raise ValueError('Current-frame K/V must share the query device and dtype')
    if any(isinstance(head, EncodingSegments) for head in heads):
        return _attend_segments(query, key, value, heads)
    descriptors, offset, longest = [], 0, 0
    for head in heads:
        pk, pv = head.key, head.value
        if pk.shape != pv.shape or pk.shape[:2] != (1, 1) or pk.shape[-1] != dim:
            raise ValueError('Each encoding head requires [1, 1, tokens, dim] K/V')
        if any(t.device != query.device or t.dtype != query.dtype for t in (pk, pv)):
            raise ValueError('Encoding K/V must share the query device and dtype')
        length = pk.shape[-2]
        descriptors.append((pk.data_ptr(), pv.data_ptr(), length, offset,
                            pk.stride(-2), pk.stride(-1), pv.stride(-2), pv.stride(-1)))
        offset += length + count
        longest = max(longest, length + count)
    from .encoding_kernels import pack_and_attend
    return pack_and_attend(query.contiguous(), key.contiguous(), value.contiguous(),
                           descriptors, offset, longest)


def _attend_segments(query, key, value, heads):
    descriptors, copies, offset, longest = [], [], 0, 0
    count = query.shape[-2]
    for h, head in enumerate(heads):
        start = offset
        parts = head.parts if isinstance(head, EncodingSegments) else ((head.key[0], head.value[0]),)
        for k, v in parts:
            if (k.shape != v.shape or k.ndim != 3 or k.shape[0] != 1 or k.shape[-1] != query.shape[-1]
                    or any(t.device != query.device or t.dtype != query.dtype for t in (k, v))):
                raise ValueError('Encoding segments must be same-device/dtype [1, tokens, dim] K/V')
            n = k.shape[-2]
            if n:
                copies.append((k.data_ptr(), v.data_ptr(), n, offset,
                               k.stride(-2), k.stride(-1), v.stride(-2), v.stride(-1)))
                offset += n
                longest = max(longest, n)
        descriptors.append((0, 0, offset-start, start, 0, 0, 0, 0))
        copies.append((key.data_ptr()+h*key.stride(1)*key.element_size(),
                       value.data_ptr()+h*value.stride(1)*value.element_size(), count, offset,
                       key.stride(-2), key.stride(-1), value.stride(-2), value.stride(-1)))
        offset += count
    from .encoding_kernels import pack_and_attend
    return pack_and_attend(query.contiguous(), key, value, descriptors, offset, max(longest, count), copies)
