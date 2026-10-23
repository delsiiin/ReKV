"""PM-KVQ grouped affine INT16 initialization and 16/8/4/2-bit packed shrinking."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


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
    def quantize(cls, value, group_size=128):
        """Initialize unsigned 16-bit codes for BOTH K and V along head_dim.

        PM-KVQ configuration: per_group, symmetric=False, round_zeros=False.
        Range arithmetic/parameters use FP32 for the FP16 ReKV model. Tail groups
        are an extension: valid elements set the range, only packed codes are padded.
        """
        if group_size < 1 or group_size % 8:
            raise ValueError('group_size must be a positive multiple of eight')
        if value.ndim != 3 or min(value.shape) < 1 or not torch.isfinite(value).all():
            raise ValueError('Expected nonempty finite [heads, tokens, channels] KV')
        x = value.detach().float()
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
        return cls(pack(codes, 16), torch.cat(scales, -1), torch.cat(offsets, -1),
                   tuple(value.shape), 16, group_size)

    def shrink(self):
        if self.bits == 2:
            return self
        n = self.bits // 2
        codes = unpack(self.data, self.bits)
        # PM-KVQ's integer bit-shift expression. int32 wraparound for 16-bit input
        # is intentional; the final mask restores unsigned n-bit codes.
        codes = (((codes << (2*n)) - (codes << n) + codes
                  + (2**(3*n-1)-2**(2*n-1)+2**(n-1))) >> (3*n)) & (2**n-1)
        return PackedKV(pack(codes, n), self.scale*(2**n+1),
                        self.offset, self.shape, n, self.group_size)

    def dequantize(self, device=None, dtype=torch.float16):
        device = self.data.device if device is None else torch.device(device)
        if device != self.data.device:
            raise ValueError('Dequantization must stay on the cache device')
        length = self.shape[-1]
        codes = unpack(self.data, self.bits)[..., :length].float()
        scale = self.scale.repeat_interleave(self.group_size, -1)[..., :length]
        offset = self.offset.repeat_interleave(self.group_size, -1)[..., :length]
        return (codes*scale-offset).to(dtype).contiguous()


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
