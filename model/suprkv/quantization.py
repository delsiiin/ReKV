"""KIVI grouping directions with PM-KVQ INT16 initialization and packed shrinking."""
from dataclasses import dataclass
import importlib.util
from numbers import Integral

import torch
import torch.nn.functional as F


def _compact_quantization_supported(value, group_size):
    return (value.is_cuda and torch.version.cuda is not None
            and value.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and group_size <= 4096 and importlib.util.find_spec('triton') is not None)


@dataclass(frozen=True)
class PackedKV:
    data: torch.Tensor
    scale: torch.Tensor
    offset: torch.Tensor
    shape: tuple
    bits: int
    group_size: int
    axis: str = 'token'

    @classmethod
    def quantize(cls, value, group_size=128, *, axis='token', bits=16, backend='auto'):
        """Initialize unsigned 16-bit codes from [heads, tokens, channels].

        'channel' fixes each channel and groups tokens (K); 'token' fixes each
        token and groups channels (V). Codes and parameters use the grouped
        dimension last; shape always describes the original input layout.
        PM-KVQ affine arithmetic: symmetric=False, round_zeros=False.
        Range arithmetic/parameters use FP32 for every model dtype. Tail groups
        are an extension: valid elements set the range, only packed codes are padded.
        """
        if axis not in ('channel', 'token'):
            raise ValueError("axis must be 'channel' or 'token'")
        if group_size < 1 or group_size % 8:
            raise ValueError('group_size must be a positive multiple of eight')
        if value.ndim != 3 or min(value.shape) < 1 or not torch.isfinite(value).all():
            raise ValueError('Expected nonempty finite [heads, tokens, channels] KV')
        if bits not in (16, 8, 4, 2):
            raise ValueError('Only 16/8/4/2 bit storage is supported')
        if backend not in ('auto', 'torch', 'triton'):
            raise ValueError('backend must be auto, torch or triton')
        use_triton = (value.is_cuda and torch.version.cuda is not None
                      and value.dtype in (torch.float16, torch.bfloat16, torch.float32)
                      and group_size <= 4096 and importlib.util.find_spec('triton') is not None)
        if backend == 'triton' and not use_triton:
            raise ValueError('Triton quantization requires CUDA FP16/BF16/FP32 and group_size <= 4096')
        if backend != 'torch' and use_triton:
            from .packed_kernels import quantize
            data, scale, offset = quantize(value, group_size, axis, bits)
            return cls(data, scale, offset, tuple(value.shape), bits, group_size, axis)
        x = value.detach().float()
        if axis == 'channel':
            x = x.transpose(1, 2)
        codes, scales, offsets = [], [], []
        for start in range(0, x.shape[-1], group_size):
            group = x[..., start:start+group_size]
            lo, hi = group.amin(-1, keepdim=True), group.amax(-1, keepdim=True)
            scale = ((hi-lo)/65535).clamp(min=1e-5, max=1e5)
            offset = -lo
            codes.append(((group+offset)/scale).round().clamp(0, 65535).to(torch.int32))
            scales.append(scale)
            offsets.append(offset)
        codes = torch.cat(codes, -1)
        # PM-KVQ packs into int16 words; 2-bit storage needs eight codes per word.
        codes = F.pad(codes, (0, (-codes.shape[-1]) % 8))
        result = cls(pack(codes, 16), torch.cat(scales, -1), torch.cat(offsets, -1),
                     tuple(value.shape), 16, group_size, axis)
        while result.bits > bits:
            result = result.shrink(backend=backend)
        return result

    def shrink(self, *, backend='auto'):
        use_triton = self._use_triton(backend, torch.float32)
        if self.bits == 2:
            return self
        n = self.bits // 2
        if use_triton:
            from .packed_kernels import shrink
            data, scale = shrink(self)
            return PackedKV(data, scale, self.offset, self.shape, n, self.group_size, self.axis)
        codes = unpack(self.data, self.bits)
        # PM-KVQ's integer bit-shift expression. int32 wraparound for 16-bit input
        # is intentional; the final mask restores unsigned n-bit codes.
        codes = (((codes << (2*n)) - (codes << n) + codes
                  + (2**(3*n-1)-2**(2*n-1)+2**(n-1))) >> (3*n)) & (2**n-1)
        return PackedKV(pack(codes, n), self.scale*(2**n+1),
                        self.offset, self.shape, n, self.group_size, self.axis)

    def _use_triton(self, backend, dtype):
        if backend not in ('auto', 'torch', 'triton'):
            raise ValueError('backend must be auto, torch or triton')
        supported = (self.data.device.type == 'cuda' and torch.version.cuda is not None
                     and dtype in (torch.float16, torch.bfloat16, torch.float32)
                     and all(x.is_contiguous() for x in (self.data, self.scale, self.offset))
                     and importlib.util.find_spec('triton') is not None)
        if backend == 'triton' and not supported:
            raise ValueError('Triton requires contiguous CUDA storage and FP16/BF16/FP32 operands')
        return backend != 'torch' and supported

    def dequantize(self, device=None, dtype=torch.float16, *, backend='auto'):
        device = self.data.device if device is None else torch.device(device)
        if device != self.data.device:
            raise ValueError('Dequantization must stay on the cache device')
        if self._use_triton(backend, dtype):
            from .packed_kernels import dequantize
            return dequantize(self, dtype)
        length = self.shape[-2] if self.axis == 'channel' else self.shape[-1]
        codes = unpack(self.data, self.bits)[..., :length].float()
        scale = self.scale.repeat_interleave(self.group_size, -1)[..., :length]
        offset = self.offset.repeat_interleave(self.group_size, -1)[..., :length]
        value = (codes*scale-offset).to(dtype)
        if self.axis == 'channel':
            value = value.transpose(1, 2)
        return value.contiguous()

    def matmul(self, left, *, transpose=False, backend='auto'):
        """left @ KV (or KV.T), decoding tiles inside the CUDA GEMM.

        left is [query_heads, rows, reduction] or [batch, query_heads,
        rows, reduction]. KV is shared across batches, with GQA heads mapped
        without repeat_interleave. The torch backend is the numerical oracle.
        """
        reduction = self.shape[2] if transpose else self.shape[1]
        if (left.ndim not in (3, 4) or not left.is_floating_point()
                or left.device != self.data.device or left.shape[-1] != reduction
                or left.shape[-3] < 1 or left.shape[-3] % self.shape[0]):
            raise ValueError('Expected same-device floating [batch?, GQA heads, rows, reduction]')
        if self._use_triton(backend, left.dtype) and not left.requires_grad:
            if left.numel() == 0:
                return left.new_empty((*left.shape[:-1], self.shape[1] if transpose else self.shape[2]))
            from .packed_kernels import matmul
            return matmul(self, left, transpose)
        dense = self.dequantize(dtype=left.dtype, backend='torch')
        dense = dense.repeat_interleave(left.shape[-3] // self.shape[0], 0)
        return left @ (dense.transpose(-1, -2) if transpose else dense)


@dataclass(frozen=True)
class CompactPackedKV:
    """Flat packed payload; every head retains its own quantization boundaries.

    Offsets are immutable host metadata, avoiding device synchronization during
    reads. Only per-row packing alignment is padded, never to the longest head.
    Head views share these three storages and are created only when needed.
    """
    data: torch.Tensor
    scale: torch.Tensor
    offset: torch.Tensor
    head_offsets: tuple
    dim: int
    bits: int
    group_size: int
    axis: str
    data_offsets: tuple
    parameter_offsets: tuple

    @classmethod
    def quantize(cls, value, head_offsets, group_size, *, axis, bits=8, backend='auto'):
        head_offsets = tuple(head_offsets)
        if (value.ndim != 2 or min(value.shape) < 1 or len(head_offsets) < 2
                or any(not isinstance(n, Integral) for n in head_offsets)
                or head_offsets[0] != 0 or head_offsets[-1] != len(value)):
            raise ValueError('Expected compact [total_head_tokens, dim] payload and offsets')
        if any(end <= start for start, end in zip(head_offsets, head_offsets[1:])):
            raise ValueError('Retained high regions must be nonempty in every head')
        if axis not in ('channel', 'token'):
            raise ValueError("axis must be 'channel' or 'token'")
        if group_size < 1 or group_size % 8:
            raise ValueError('group_size must be a positive multiple of eight')
        if bits not in (16, 8, 4, 2):
            raise ValueError('Only 16/8/4/2 bit storage is supported')
        if backend not in ('auto', 'torch', 'triton'):
            raise ValueError('backend must be auto, torch or triton')
        use_triton = _compact_quantization_supported(value, group_size)
        if backend == 'triton' and not use_triton:
            raise ValueError('Triton quantization requires CUDA FP16/BF16/FP32 and group_size <= 4096')
        if backend != 'torch' and use_triton:
            from .packed_kernels import quantize_compact
            data, scale, offset, data_offsets, parameter_offsets = quantize_compact(
                value, head_offsets, group_size, axis, bits)
            return cls(data, scale, offset, head_offsets, value.shape[-1], bits, group_size, axis,
                       data_offsets, parameter_offsets)
        parts, data_offsets, parameter_offsets = [], [0], [0]
        for start, end in zip(head_offsets, head_offsets[1:]):
            part = PackedKV.quantize(value[start:end][None], group_size, axis=axis, bits=bits, backend=backend)
            parts.append(part)
            data_offsets.append(data_offsets[-1] + part.data.numel())
            parameter_offsets.append(parameter_offsets[-1] + part.scale.numel())
        return cls(*(torch.cat([getattr(p, name).flatten() for p in parts]) for name in ('data', 'scale', 'offset')),
                   tuple(head_offsets), value.shape[-1], bits, group_size, axis,
                   tuple(data_offsets), tuple(parameter_offsets))

    def head(self, index):
        tokens = self.head_offsets[index+1] - self.head_offsets[index]
        rows, length = (self.dim, tokens) if self.axis == 'channel' else (tokens, self.dim)
        words = ((length + 7) // 8 * 8) // (16 // self.bits)
        groups = (length + self.group_size - 1) // self.group_size
        begin, end = self.data_offsets[index:index+2]
        pbegin, pend = self.parameter_offsets[index:index+2]
        return PackedKV(self.data[begin:end].view(1, rows, words),
                        self.scale[pbegin:pend].view(1, rows, groups),
                        self.offset[pbegin:pend].view(1, rows, groups),
                        (1, tokens, self.dim), self.bits, self.group_size, self.axis)

    def shrink(self, *, backend='auto'):
        if self.bits == 2:
            return self
        # Each packed row is aligned for all supported precisions, so the
        # existing integer lowering kernel can process the flat storage once.
        flat = PackedKV(self.data, self.scale, self.offset, (), self.bits, self.group_size, self.axis)
        lowered = flat.shrink(backend=backend)
        return CompactPackedKV(lowered.data, lowered.scale, lowered.offset,
                               self.head_offsets, self.dim, lowered.bits, self.group_size, self.axis,
                               tuple(x // 2 for x in self.data_offsets), self.parameter_offsets)

    def dequantize(self, device=None, dtype=torch.float16, *, backend='auto'):
        return torch.cat([self.head(h).dequantize(device, dtype, backend=backend)[0]
                          for h in range(len(self.head_offsets)-1)])


def pack(codes, bits):
    """PM-KVQ int16 containers, first code in the most significant bits."""
    if bits not in (16, 8, 4, 2):
        raise ValueError('Only 16/8/4/2 bit storage is supported')
    per_word = 16 // bits
    if codes.shape[-1] % per_word:
        raise ValueError('Unaligned quantization group')
    x = codes.to(torch.int32).reshape(*codes.shape[:-1], -1, per_word)
    shifts = torch.arange(per_word-1, -1, -1, device=x.device, dtype=torch.int32)*bits
    return (x << shifts).sum(-1).to(torch.int16).contiguous()


def unpack(data, bits):
    if bits not in (16, 8, 4, 2):
        raise ValueError('Only 16/8/4/2 bit storage is supported')
    per_word = 16 // bits
    shifts = torch.arange(per_word-1, -1, -1, device=data.device, dtype=torch.int32)*bits
    return ((data.to(torch.int32).unsqueeze(-1) >> shifts) & ((1 << bits)-1)).flatten(-2)
