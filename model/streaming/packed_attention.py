"""Question-local packed KV, retaining the snapshot's immutable GPU payloads."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .quantization import PackedKV
from .rotary import rotate_at


@dataclass(frozen=True)
class AttentionChunk:
    key: object
    value: object
    ranks: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor

    @classmethod
    def create(cls, key, value, ranks, stored_positions, rope):
        delta = ranks - stored_positions
        angles = delta.float()[:, None] * rope.distance_scale * rope.inv_freq.to(ranks.device)
        if not isinstance(key, PackedKV):
            key = rotate_at(key, delta, rope)
            value = value.contiguous()
        return cls(key, value, ranks.contiguous(), angles.cos().contiguous(), angles.sin().contiguous())


@dataclass(frozen=True)
class PackedRotaryCache:
    chunks: tuple
    length: int
    dtype: torch.dtype
    # Generated tokens share one dense tail, instead of one kernel per token.
    generated: int = 0

    def append(self, key, value, rope):
        count = key.shape[-2]
        ranks = torch.arange(self.length, self.length + count, device=key.device)
        # New keys already use their logical positions. Dense chunks do not
        # read rotation tables; reuse a buffer instead of launching trig ops.
        dummy = key.new_empty((0, rope.inv_freq.numel()))
        chunk = AttentionChunk(key[0].contiguous(), value[0].contiguous(), ranks, dummy, dummy)
        chunks = self.chunks
        if self.generated:
            tail = chunks[-1]
            chunk = AttentionChunk(torch.cat((tail.key, chunk.key), 1),
                                   torch.cat((tail.value, chunk.value), 1),
                                   torch.cat((tail.ranks, ranks)),
                                   dummy, dummy)
            chunks = chunks[:-1]
        return PackedRotaryCache((*chunks, chunk), self.length + count, self.dtype, self.generated + count)

    def materialize(self):
        """Fallback for window rollover; all returned keys are rotated at ranks."""
        keys, values, ranks = [], [], []
        for chunk in self.chunks:
            k, v = chunk.key, chunk.value
            if isinstance(k, PackedKV):
                k = k.dequantize(dtype=self.dtype)
                half = chunk.cos.shape[-1]
                part = k[..., :2*half].float()
                a, b = part.chunk(2, -1)
                cos = torch.cat((chunk.cos, chunk.cos), -1)
                sin = torch.cat((chunk.sin, chunk.sin), -1)
                k = torch.cat(((part*cos + torch.cat((-b, a), -1)*sin).to(self.dtype),
                               k[..., 2*half:]), -1)
                v = v.dequantize(dtype=self.dtype)
            keys.append(k)
            values.append(v)
            ranks.append(chunk.ranks)
        ranks = torch.cat(ranks)
        # Coalesced K groups have masked padding between original frames.
        order = ranks.argsort()[:self.length]
        return (torch.cat(keys, 1)[:, order][None].contiguous(),
                torch.cat(values, 1)[:, order][None].contiguous(), ranks[order])

    def attend(self, query, rope):
        from .packed_kernels import attention_chunks
        past = self.length - query.shape[-2]
        q = rotate_at(query, torch.arange(past, self.length, device=query.device), rope)
        return attention_chunks(q, self.chunks, past)


def coalesce(chunks):
    """Batch equal formats once per question, without decoding/requantizing.

    Pad K along tokens to complete each frame's last quantization group. V and
    ranks receive matching masked rows. Parameters remain byte-for-byte intact.
    This trades a question-local packed copy for fewer per-token launches.
    """
    groups = {}
    for chunk in chunks:
        k = chunk.key
        key = (k.bits, k.group_size) if isinstance(k, PackedKV) else (0, 0)
        groups.setdefault(key, []).append(chunk)
    merged = []
    for (bits, group), parts in groups.items():
        if len(parts) == 1:
            merged.append(parts[0])
            continue
        keys, values, ranks, cos, sin = [], [], [], [], []
        for part in parts:
            k, v = part.key, part.value
            padding = (-k.shape[1]) % group if bits else 0
            if bits and padding:
                h, t, d = k.shape
                words = (t + padding) // (16 // bits)
                k = PackedKV(F.pad(k.data, (0, words-k.data.shape[-1])), k.scale, k.offset,
                             (h, t+padding, d), bits, group, 'channel')
                v = PackedKV(F.pad(v.data, (0, 0, 0, padding)),
                             F.pad(v.scale, (0, 0, 0, padding)), F.pad(v.offset, (0, 0, 0, padding)),
                             (h, t+padding, d), bits, group, 'token')
            keys.append(k)
            values.append(v)
            # F.pad's constant is converted through a float; INT64_MAX rounds
            # out of range and can become negative, accidentally unmasking it.
            ranks.append(F.pad(part.ranks, (0, padding), value=torch.iinfo(torch.int32).max) if padding else part.ranks)
            cos.append(F.pad(part.cos, (0, 0, 0, padding)) if padding else part.cos)
            sin.append(F.pad(part.sin, (0, 0, 0, padding)) if padding else part.sin)
        if bits:
            h, _, d = keys[0].shape
            shape = (h, sum(k.shape[1] for k in keys), d)
            def combine(payloads, axis):
                dim = 2 if axis == 'channel' else 1
                return PackedKV(*(torch.cat([getattr(p, name) for p in payloads], dim)
                                  for name in ('data', 'scale', 'offset')),
                                shape, bits, group, axis)
            key, value = combine(keys, 'channel'), combine(values, 'token')
        else:
            key, value = torch.cat(keys, 1), torch.cat(values, 1)
        merged.append(AttentionChunk(key, value, torch.cat(ranks), torch.cat(cos), torch.cat(sin)))
    return tuple(merged)


def from_frames(sink, frames, layer, rope, dtype):
    key, value = sink
    ranks = torch.arange(key.shape[1], device=key.device)
    chunks = [AttentionChunk.create(key, value, ranks, ranks, rope)] if ranks.numel() else []
    length = key.shape[1]
    for frame in frames:
        entry = frame.layers[layer]
        indices = torch.cat((entry.high_indices, entry.low_indices))
        # high/low partitions are not chronological; ranks preserve the same
        # causal positions as materialize(). Storage itself need not be sorted.
        ranks = indices.argsort().argsort() + length
        high = entry.high_indices.numel()
        for k, v, ids, rank in (
                (entry.high_k, entry.high_v, entry.high_indices, ranks[:high]),
                (entry.low_k, entry.low_v, entry.low_indices, ranks[high:])):
            if ids.numel():
                chunks.append(AttentionChunk.create(k, v, rank, ids.long()+frame.rope_start, rope))
        length += indices.numel()
    return PackedRotaryCache(coalesce(chunks), length, dtype)
