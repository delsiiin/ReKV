"""Single-writer immutable frame cache with optional CPU storage for evicted frames."""
from dataclasses import dataclass, is_dataclass, replace
import math
import importlib.util
import sys
import time

import torch

from .quantization import PackedKV, CompactPackedKV
from .segments import summary_scores, segment_summary_scores, segment_frame_query_scores, frame_local_query_scores
from .profiling import measure
from .accounting import MemoryLedger, TrackedDict
from .rotary import rotate_at


def footprint(value, seen=None, storages=None):
    """Python metadata plus actual tensor storage, deduplicating aliases."""
    seen = set() if seen is None else seen
    storages = set() if storages is None else storages
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        identity = (str(value.device), storage.data_ptr(), storage.nbytes())
        if identity not in storages:
            storages.add(identity)
            size += storage.nbytes()
    elif is_dataclass(value):
        size += footprint(vars(value), seen, storages)
    elif isinstance(value, dict):
        size += sum(footprint(k, seen, storages) + footprint(v, seen, storages) for k, v in value.items())
    elif isinstance(value, (tuple, list, set)):
        size += sum(footprint(v, seen, storages) for v in value)
    return size


def tensor_storage_bytes(value, device_type=None, seen=None, storages=None):
    """Unique tensor payload bytes, excluding Python metadata and allocator reserves."""
    seen = set() if seen is None else seen
    storages = set() if storages is None else storages
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        identity = (str(value.device), storage.data_ptr(), storage.nbytes())
        if identity in storages or (device_type is not None and value.device.type != device_type):
            return 0
        storages.add(identity)
        return storage.nbytes()
    if is_dataclass(value):
        value = vars(value)
    children = value.values() if isinstance(value, dict) else value if isinstance(value, (tuple, list)) else ()
    return sum(tensor_storage_bytes(child, device_type, seen, storages) for child in children)


def resident_copy(x, dtype=torch.float16):
    return x.detach().to(dtype=dtype).clone().contiguous()


def _fused_partition_supported(entries, scores):
    return (scores.is_cuda and torch.version.cuda is not None and scores.dtype == torch.float32
            and 0 < scores.shape[-1] <= 4096 and not scores.requires_grad
            and importlib.util.find_spec('triton') is not None
            and all(entry.high_k.dtype in (torch.float16, torch.bfloat16, torch.float32)
                    and entry.high_k.dtype == entries[0].high_k.dtype
                    and entry.high_v.shape == entry.high_k.shape
                    and entry.high_k.shape == entries[0].high_k.shape
                    and entry.high_indices.shape == (scores.shape[-1],)
                    and entry.high_indices.dtype == torch.int32
                    and all(x.device == scores.device and x.is_contiguous() and not x.requires_grad
                            for x in (entry.high_k, entry.high_v, entry.high_indices))
                    and entry.high_v.dtype == entry.high_k.dtype for entry in entries))


@dataclass(frozen=True)
class LayerFrame:
    high_indices: torch.Tensor
    high_k: torch.Tensor | PackedKV | CompactPackedKV
    high_v: torch.Tensor | PackedKV | CompactPackedKV
    low_indices: torch.Tensor
    low_k: object
    low_v: object
    representative: torch.Tensor
    bits: int = 16  # 16 denotes unquantized model dtype; otherwise integer code bits.
    dtype: torch.dtype = torch.float16
    # None denotes the rectangular, unclassified/legacy layout. Classified
    # payloads are [sum(head_lengths), D], with head-major flat token indices.
    high_offsets: tuple | None = None
    low_offsets: tuple | None = None

    @classmethod
    def full(cls, key, value, dtype=torch.float16):
        key, value = resident_copy(key, dtype), resident_copy(value, dtype)
        if key.shape != value.shape or key.ndim != 3:
            raise ValueError('Expected matching [kv_heads, tokens, head_dim] tensors')
        return cls(torch.arange(key.shape[1], dtype=torch.int32, device=key.device), key, value,
                   torch.empty(0, dtype=torch.int32, device=key.device), None, None,
                   key.float().mean(1).to(dtype), dtype=dtype)

    def classify(self, query, score_key, *, headwise=False):
        """Use pre-RoPE scores to select rows from the post-RoPE KV payload."""
        if score_key.shape != self.high_k.shape:
            raise ValueError('Scoring key must match the unpartitioned frame')
        return self.classify_scores(summary_scores(query, score_key, headwise=headwise))

    def classify_scores(self, scores):
        """Partition each head at its frame-local mean; payloads own storage."""
        if scores.ndim == 2:
            if scores.shape != self.high_k.shape[:2]:
                raise ValueError('Expected one score per KV head and token')
            high = scores >= scores.mean(-1, keepdim=True)
            counts = high.sum(-1).tolist()  # One synchronization per frame, not per head.
            high_offsets, low_offsets = [0], [0]
            for count in counts:
                high_offsets.append(high_offsets[-1] + count)
                low_offsets.append(low_offsets[-1] + scores.shape[-1] - count)
            high_rows = high.flatten().nonzero(as_tuple=True)[0]
            low_rows = (~high).flatten().nonzero(as_tuple=True)[0]
            indices = self.high_indices.expand(scores.shape[0], -1).reshape(-1)
            key, value = self.high_k.flatten(0, 1), self.high_v.flatten(0, 1)
            return replace(self, high_indices=indices.index_select(0, high_rows),
                           low_indices=indices.index_select(0, low_rows),
                           high_k=key.index_select(0, high_rows), high_v=value.index_select(0, high_rows),
                           low_k=key.index_select(0, low_rows), low_v=value.index_select(0, low_rows),
                           high_offsets=tuple(high_offsets), low_offsets=tuple(low_offsets))
        if scores.ndim != 1 or scores.shape[0] != self.high_k.shape[1]:
            raise ValueError('Expected per-head scores or an explicit shared token mask')
        high = scores >= scores.mean()
        # Boolean indexing would rediscover these positions for all six payloads.
        high_rows = high.nonzero(as_tuple=True)[0]
        low_rows = (~high).nonzero(as_tuple=True)[0]
        return replace(self, high_indices=self.high_indices.index_select(0, high_rows),
                       high_k=self.high_k.index_select(1, high_rows),
                       high_v=self.high_v.index_select(1, high_rows),
                       low_indices=self.high_indices.index_select(0, low_rows),
                       low_k=self.high_k.index_select(1, low_rows),
                       low_v=self.high_v.index_select(1, low_rows))

    @staticmethod
    def classify_batch(entries, scores):
        """Partition a layer's unclassified frames, with one CUDA counts readback."""
        if not entries or scores.ndim != 3 or scores.shape[0] != len(entries):
            raise ValueError('Expected [frames, KV heads, tokens] scores')
        if any(entry.high_offsets is not None or entry.high_k.shape[:2] != scores.shape[1:]
               for entry in entries):
            raise ValueError('Scores must match unpartitioned frame KV')
        if _fused_partition_supported(entries, scores):
            from .scoring_kernels import partition_frames
            return [replace(entry, **parts) for entry, parts in zip(entries, partition_frames(entries, scores))]
        return [entry.classify_scores(row) for entry, row in zip(entries, scores)]

    def shrink(self, group_size):
        """Lower the retained high region by one precision level."""
        if not self.high_indices.numel() or self.bits == 2:
            return self
        if self.bits == 16:
            if self.high_offsets is not None:
                key = CompactPackedKV.quantize(self.high_k, self.high_offsets, group_size, axis='channel')
                value = CompactPackedKV.quantize(self.high_v, self.high_offsets, group_size, axis='token')
            else:
                key = PackedKV.quantize(self.high_k, group_size, axis='channel', bits=8)
                value = PackedKV.quantize(self.high_v, group_size, axis='token', bits=8)
        else:
            key, value = self.high_k.shrink(), self.high_v.shrink()
        return replace(self, high_k=key, high_v=value, bits=key.bits)

    def drop_low(self):
        if not self.low_indices.numel():
            return self
        return replace(self, low_indices=torch.empty(0, dtype=torch.int32, device=self.high_indices.device),
                       low_k=None, low_v=None,
                       low_offsets=None if self.low_offsets is None else (0,) * len(self.low_offsets))

    def to(self, device, *, pin_memory=False):
        """Move compressed payloads intact, including indices and quantizer metadata.

        D2H is blocking so the archive is ready before publication. Pinned CPU
        payloads can be loaded asynchronously on the retrieval stream.
        """
        device = torch.device(device)
        if self.high_indices.device == device:
            return self

        def move(value):
            if isinstance(value, torch.Tensor):
                result = value.to(device=device, non_blocking=device.type == 'cuda')
                return result.pin_memory() if pin_memory and device.type == 'cpu' else result
            if isinstance(value, (PackedKV, CompactPackedKV)):
                return replace(value, data=move(value.data), scale=move(value.scale), offset=move(value.offset))
            return value

        return replace(self, **{name: move(getattr(self, name)) for name in
                       ('high_indices', 'high_k', 'high_v', 'low_indices', 'low_k', 'low_v', 'representative')})

    @property
    def heads(self):
        return self.representative.shape[0]

    def head_parts(self, head):
        """Views of high/low payloads for one KV head, without decoding."""
        for key, value, ids, offsets in ((self.high_k, self.high_v, self.high_indices, self.high_offsets),
                                         (self.low_k, self.low_v, self.low_indices, self.low_offsets)):
            if offsets is not None:
                start, end = offsets[head:head+2]
                if end == start:
                    continue
                def view(payload):
                    return payload.head(head) if isinstance(payload, CompactPackedKV) else payload[start:end][None]
                yield view(key), view(value), ids[start:end]
            elif ids.numel():
                def view(payload):
                    if isinstance(payload, PackedKV):
                        return replace(payload, data=payload.data[head:head+1], scale=payload.scale[head:head+1],
                                       offset=payload.offset[head:head+1], shape=(1, *payload.shape[1:]))
                    return payload[head:head+1]
                yield view(key), view(value), ids

    def materialize_head(self, head):
        keys, values, indices = [], [], []
        for key, value, ids in self.head_parts(head):
            keys.append(key.dequantize(dtype=self.dtype) if isinstance(key, PackedKV) else key)
            values.append(value.dequantize(dtype=self.dtype) if isinstance(value, PackedKV) else value)
            indices.append(ids)
        indices = torch.cat(indices)
        order = indices.argsort()
        return torch.cat(keys, 1)[:, order].contiguous(), torch.cat(values, 1)[:, order].contiguous(), indices[order]

    def materialize(self, device=None):
        device = self.high_indices.device if device is None else torch.device(device)
        if device != self.high_indices.device:
            raise ValueError('Cache materialization must stay on the cache device; offload is disabled')
        if self.high_offsets is not None:
            parts = [self.materialize_head(h) for h in range(self.heads)]
            lengths = [part[0].shape[1] for part in parts]
            if len(set(lengths)) != 1:
                raise ValueError('Unequal head lengths require materialize_head() or headwise retrieval')
            # Recent frames retain both regions and therefore identical source
            # positions. Other equal-length heads can have distinct positions.
            ids = torch.stack([part[2] for part in parts])
            if torch.equal(ids, ids[:1].expand_as(ids)):
                ids = ids[0]
            return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]), ids
        keys = [self.high_k.dequantize(device, dtype=self.dtype) if isinstance(self.high_k, PackedKV) else self.high_k.to(device)]
        values = [self.high_v.dequantize(device, dtype=self.dtype) if isinstance(self.high_v, PackedKV) else self.high_v.to(device)]
        indices = [self.high_indices]
        if self.low_indices.numel():
            keys.append(self.low_k.dequantize(device, dtype=self.dtype) if isinstance(self.low_k, PackedKV) else self.low_k.to(device))
            values.append(self.low_v.dequantize(device, dtype=self.dtype) if isinstance(self.low_v, PackedKV) else self.low_v.to(device))
            indices.append(self.low_indices)
        indices = torch.cat(indices)
        order = indices.argsort().to(device)
        return (torch.cat(keys, 1)[:, order].contiguous(),
                torch.cat(values, 1)[:, order].contiguous(), indices.sort().values)


@dataclass(frozen=True)
class Frame:
    frame_id: int
    timestamp: float
    segment_id: int
    layers: tuple
    closed: bool = False
    hits: int = 0  # Number of questions that retrieved this frame (once per question).
    score_error: str = ''
    rope_start: int = 0
    token_count: int = 0


@dataclass(frozen=True)
class CacheSnapshot:
    version: int
    question_id: int
    sink: tuple
    frames: tuple
    recent_ids: tuple

    def retrieve_layer(self, layer, query, topk=64, profiler=None, *,
                       position_bias=None, with_rope_positions=False, packed=False, headwise=False,
                       batched=False):
        """Select frames; headwise=True returns (HeadwiseRotaryCache, frame_ids).

        The rectangular compatibility result requires equal total head lengths.
        Its source positions may be [heads, tokens] when indices differ.
        """
        if topk < len(self.recent_ids):
            raise ValueError('topk must accommodate the recent frames')
        protected = set(self.recent_ids)
        available = [f for f in self.frames if f.frame_id not in protected]
        if query.device != self.sink[layer][0].device:
            raise ValueError('Query and resident cache must be on the same device')
        q = query.detach().float().mean(dim=-2).squeeze(0)
        if available:
            # Transfer only the small retrieval vectors before ranking. Group by
            # device to avoid one H2D transfer per archived frame.
            available.sort(key=lambda f: str(f.layers[layer].representative.device))
            groups = {}
            for frame in available:
                representative = frame.layers[layer].representative
                groups.setdefault(representative.device, []).append(representative)
            parts = [torch.stack(rows).to(query.device) for rows in groups.values()]
            keys = (parts[0] if len(parts) == 1 else torch.cat(parts)).float()
            if q.shape[0] % keys.shape[1]:
                raise ValueError('Invalid GQA heads')
            keys = keys.repeat_interleave(q.shape[0] // keys.shape[1], 1)
            if position_bias is not None:
                # Representatives come from post-RoPE keys. Score each frame with
                # Q placed immediately after that frame in its stored coordinates.
                offsets = torch.tensor([f.rope_start+f.token_count for f in available], device=query.device)
                local_q = rotate_at(query.float(), torch.arange(query.shape[-2], device=query.device),
                                    position_bias).mean(-2, keepdim=True)
                # Rotation is linear: pool once, then shift each frame's query.
                # Avoid an [frames, heads, question_tokens, dim] temporary.
                q = rotate_at(local_q.expand(len(available), -1, -1, -1),
                              offsets[:, None, None], position_bias).squeeze(-2)
            scores = (keys*q).sum((1, 2))
            ranking = scores.argsort(descending=True, stable=True).tolist()
            available = [available[i] for i in ranking]
        selected = [f for f in self.frames if f.frame_id in protected] + available[:topk-len(protected)]
        selected.sort(key=lambda f: (f.timestamp, f.frame_id))
        device = query.device
        with measure(profiler, 'cache_materialization'):
            # Only selected frames and this decoder layer are loaded. The CPU
            # archive remains authoritative; retrieval does not grow residency.
            selected = [replace(frame, layers=frame.layers[:layer] +
                                (frame.layers[layer].to(device),) + frame.layers[layer+1:])
                        if frame.layers[layer].high_indices.device != device else frame for frame in selected]
            if batched:
                if position_bias is None or packed:
                    raise ValueError('Batched dense retrieval requires RoPE and packed=False')
                from .ragged_cache import from_frames
                return (from_frames(self.sink[layer], selected, layer, position_bias, query.dtype),
                        tuple(f.frame_id for f in selected))
            compact = any(f.layers[layer].high_offsets is not None for f in selected)
            if headwise or compact:
                from .attention import RotaryCache
                from .packed_attention import HeadwiseRotaryCache, from_frames
                heads = []
                frame_indices = [[] for _ in selected]
                sink_k, sink_v = self.sink[layer]
                for head in range(sink_k.shape[0]):
                    if packed:
                        if position_bias is None:
                            raise ValueError('Packed retrieval requires explicit RoPE')
                        heads.append(from_frames(self.sink[layer], selected, layer, position_bias, query.dtype, head=head))
                    else:
                        parts = [f.layers[layer].materialize_head(head) for f in selected]
                        for indices, part in zip(frame_indices, parts):
                            indices.append(part[2])
                        keys = torch.cat([sink_k[head:head+1]] + [p[0] for p in parts], 1)[None]
                        values = torch.cat([sink_v[head:head+1]] + [p[1] for p in parts], 1)[None]
                        positions = torch.cat([torch.arange(sink_k.shape[1], device=device)] +
                                              [p[2].long()+f.rope_start for f, p in zip(selected, parts)])
                        heads.append(RotaryCache(keys, values, positions))
                ids = tuple(f.frame_id for f in selected)
                if headwise or packed:
                    return HeadwiseRotaryCache(tuple(heads)), ids
                if len({h.key.shape[-2] for h in heads}) != 1:
                    raise ValueError('Unequal head lengths require headwise=True')
                positions = torch.stack([h.positions for h in heads])
                if torch.equal(positions, positions[:1].expand_as(positions)):
                    positions = positions[0]
                metadata = positions if with_rope_positions else tuple(
                    (f.frame_id, tuple(indices)) for f, indices in zip(selected, frame_indices))
                return torch.cat([h.key for h in heads], 1), torch.cat([h.value for h in heads], 1), ids, metadata
            if packed:
                if position_bias is None:
                    raise ValueError('Packed retrieval requires explicit RoPE')
                from .packed_attention import from_frames
                return (from_frames(self.sink[layer], selected, layer, position_bias, query.dtype),
                        tuple(f.frame_id for f in selected))
            keys, values = [self.sink[layer][0].to(device)], [self.sink[layer][1].to(device)]
            positions = []
            rope_positions = [torch.arange(keys[0].shape[1], device=device)] if with_rope_positions else None
            for frame in selected:
                key, value, ids = frame.layers[layer].materialize(device)
                keys.append(key)
                values.append(value)
                positions.append((frame.frame_id, ids))
                if with_rope_positions:
                    rope_positions.append(ids.long()+frame.rope_start)
            return (torch.cat(keys, 1)[None], torch.cat(values, 1)[None],
                    tuple(f.frame_id for f in selected),
                    torch.cat(rope_positions) if with_rope_positions else positions)


class StreamingKVCache:
    def __init__(self, sink, budget_tokens, frame_tokens, recent_frames=2, group_size=128, profiler=None,
                 dtype=torch.float16, compression_granularity='layer', scoring_scope='segment', offload_enabled=False):
        self.profiler = profiler
        if type(offload_enabled) is not bool:
            raise ValueError('offload_enabled must be a boolean')
        self.offload_enabled = offload_enabled
        if compression_granularity not in ('layer', 'head'):
            raise ValueError('Invalid compression_granularity')
        self.compression_granularity = compression_granularity
        if scoring_scope not in ('segment', 'frame'):
            raise ValueError('Invalid scoring_scope')
        self.scoring_scope = scoring_scope
        if budget_tokens <= 0 or recent_frames < 1 or frame_tokens < 1:
            raise ValueError('Budget, recent_frames and frame_tokens must be positive')
        if group_size < 1 or group_size % 8:
            raise ValueError('group_size must be a positive multiple of eight')
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError('Cache dtype must be float16, bfloat16 or float32')
        self.dtype = dtype
        self.sink = tuple((resident_copy(k, dtype), resident_copy(v, dtype)) for k, v in sink)
        if not self.sink:
            raise ValueError('At least one layer is required')
        self.device = self.sink[0][0].device
        if any(k.device != self.device or v.device != self.device for k, v in self.sink):
            raise ValueError('All cache layers must use one device')
        self.budget = budget_tokens * 4 * sum(k.shape[0]*k.shape[-1] for k, _ in self.sink)
        self.frame_tokens, self.recent_frames, self.group_size = frame_tokens, recent_frames, group_size
        self._frames = TrackedDict(changed=self._frame_changed)
        self.offloaded_frames = TrackedDict(changed=self._offloaded_changed)
        # Pre-RoPE scoring keys are transient work memory, never snapshot KV.
        self._score_keys = TrackedDict(changed=self._diagnostics_changed)
        self._score_queries = TrackedDict(changed=self._diagnostics_changed)
        self.next_id = self.segment_id = self.version = 0
        self.last_timestamp = float('-inf')
        self.last_activation_id = -1
        self.last_snapshot_id = -1
        self.snapshots = {}
        self.counts = dict(compressions=0, dropped_low_frames=0, evicted_frames=0, offloaded_frames=0, scoring_failures=0)
        self.last_event = {}
        # Conservative protected-region reserve, including split-frame tensor metadata.
        layers = tuple(LayerFrame.full(torch.zeros(k.shape[0], frame_tokens, k.shape[-1], dtype=dtype),
                                      torch.zeros(k.shape[0], frame_tokens, k.shape[-1], dtype=dtype), dtype)
                       for k, _ in self.sink)
        if self.compression_granularity == 'head':
            layers = tuple(x.classify_scores(torch.zeros(x.heads, frame_tokens)) for x in layers)
        reserve = tuple(Frame(i, float(i), 0, tuple(replace(x,
            low_k=torch.empty((*x.high_k.shape[:-2], 0, x.high_k.shape[-1]), dtype=dtype),
            low_v=torch.empty((*x.high_v.shape[:-2], 0, x.high_v.shape[-1]), dtype=dtype))
            for x in layers), True) for i in range(recent_frames))
        # Each recent frame owns separate storage; the prototype above shares it.
        protected = footprint((self.sink, reserve)) + (recent_frames-1)*footprint(layers) + 4096
        if self.budget < protected:
            raise ValueError(f'Budget {self.budget} bytes cannot hold sink/recent metadata; need at least {protected}')
        self._reset_accounting()

    @property
    def frames(self):
        return self._frames

    @frames.setter
    def frames(self, value):
        if value is self._frames:
            return
        self._frames = TrackedDict(value, changed=self._frame_changed)
        for frame_id in tuple(self._score_keys):
            if frame_id not in self._frames or self._frames[frame_id].closed:
                self._score_keys.pop(frame_id)
        for frame_id in tuple(self._score_queries):
            if frame_id not in self._frames or self._frames[frame_id].closed:
                self._score_queries.pop(frame_id)
        # Bulk replacement is a cold compatibility/debug path. Normal updates
        # use the tracked table and never rebuild the object graph.
        self._reset_accounting()

    @property
    def counts(self):
        return self._counts

    @counts.setter
    def counts(self, value):
        if value is not getattr(self, '_counts', None):
            self._counts = TrackedDict(value, changed=self._diagnostics_changed)

    @property
    def snapshots(self):
        return self._snapshots

    @snapshots.setter
    def snapshots(self, value):
        if value is not getattr(self, '_snapshots', None):
            self._snapshots = TrackedDict(value, changed=self._snapshot_changed)

    @property
    def last_event(self):
        return self._last_event

    @last_event.setter
    def last_event(self, value):
        if value is not getattr(self, '_last_event', None):
            self._last_event = TrackedDict(value, changed=self._diagnostics_changed)

    def _diagnostics_changed(self, mapping, key, previous, value):
        self._working_ledger.mapping_changed(mapping, key)

    def _offloaded_changed(self, mapping, key, previous, value):
        self._offload_ledger.mapping_changed(mapping, key)
        self._working_ledger.mapping_changed(mapping, key)
        if self.offload_enabled:
            self._cache_ledger.mapping_changed(mapping, key)

    def _reset_accounting(self):
        self._recent_key = None
        recent = set(self.recent_ids)
        self._accounted_recent = recent
        self._accounted_recent_ids = self.recent_ids
        self._managed = {i: f for i, f in self.frames.items() if f.closed or i in recent}
        self._open_ids = dict.fromkeys(i for i, f in self.frames.items() if not f.closed)
        self._bits = {str(b): 0 for b in (16, 8, 4, 2, 0)}
        self._frame_counts = {}
        for frame in self.frames.values():
            for layer in frame.layers:
                self._bits[str(layer.bits)] += 1
            self._frame_counts[frame.frame_id] = self._token_counts(frame)
        self._persistent_ledger, self._working_ledger = MemoryLedger(), MemoryLedger()
        self._persistent_root = self._persistent_value()
        self._offload_ledger = MemoryLedger()
        self._offload_ledger.add(self.offloaded_frames)
        self._cache_root = (self._persistent_root, self.offloaded_frames) if self.offload_enabled else self._persistent_root
        self._cache_ledger = MemoryLedger() if self.offload_enabled else self._persistent_ledger
        if self.offload_enabled:
            self._cache_ledger.add(self._cache_root)
        self._working_root = (self._cache_root, self.frames, self.snapshots, self.counts, self.last_event,
                              self._score_keys, self._score_queries)
        self._persistent_ledger.add(self._persistent_root)
        self._working_ledger.add(self._working_root)

    def _frame_changed(self, mapping, key, previous, value):
        if value is None or value.closed:
            self._score_keys.pop(key, None)
            self._score_queries.pop(key, None)
        # Appends and eviction of non-recent frames only need the previous top-k
        # IDs. Avoid sorting every retained ID on every insertion/deletion.
        if (self._recent_key is not None and self._recent_key[0] == id(mapping)
                and self._recent_key[2] == self.recent_frames):
            if previous is None:
                self._recent_ids = tuple(sorted((*self._recent_ids, key))[-self.recent_frames:])
                self._recent_key = (id(mapping), mapping.key_version, self.recent_frames)
            elif value is None and key not in self._recent_ids:
                self._recent_key = (id(mapping), mapping.key_version, self.recent_frames)
        self._working_ledger.mapping_changed(mapping, key)
        if previous is None or value is None or previous.layers is not value.layers:
            if previous is not None:
                for layer in previous.layers:
                    self._bits[str(layer.bits)] -= 1
            if value is not None:
                for layer in value.layers:
                    self._bits[str(layer.bits)] += 1
                self._frame_counts[key] = self._token_counts(value)
            else:
                self._frame_counts.pop(key, None)
        if value is None or value.closed:
            self._open_ids.pop(key, None)
        else:
            self._open_ids[key] = None
        self._update_managed(key)

    @staticmethod
    def _token_counts(frame):
        counts = []
        for layer in frame.layers:
            row = [layer.low_indices.numel(), 0, 0, 0]
            row[{16: 0, 8: 1, 4: 2, 2: 3}[layer.bits]] += layer.high_indices.numel()
            if layer.high_offsets is not None:
                row = [n / layer.heads for n in row]
            counts.append(tuple(row))
        return tuple(counts)

    def _update_managed(self, changed=None):
        recent_ids = self.recent_ids
        if changed is None and recent_ids == self._accounted_recent_ids:
            return
        recent = set(recent_ids)
        affected = recent ^ self._accounted_recent
        if changed is not None:
            affected.add(changed)
        for key in affected:
            frame = self.frames.get(key)
            if frame is not None and (frame.closed or key in recent):
                self._managed[key] = frame
            else:
                self._managed.pop(key, None)
            self._persistent_ledger.mapping_changed(self._managed, key)
            self._working_ledger.mapping_changed(self._managed, key)
            if self.offload_enabled:
                self._cache_ledger.mapping_changed(self._managed, key)
        self._accounted_recent = recent
        self._accounted_recent_ids = recent_ids

    def _snapshot_changed(self, mapping, key, previous, value):
        self._working_ledger.mapping_changed(mapping, key)

    def _persistent_value(self):
        return (self.sink, self._managed, self.budget, self.frame_tokens, self.recent_frames,
                self.group_size, self.next_id, self.segment_id, self.version, self.last_activation_id)

    def _refresh_accounting(self):
        # recent_frames and scalar configuration can also be adjusted by callers.
        self._update_managed()
        persistent = self._persistent_value()
        if all(a is b for a, b in zip(persistent, self._persistent_root)):
            persistent = self._persistent_root
        else:
            self._persistent_ledger.add(persistent)
            self._persistent_ledger.remove(id(self._persistent_root))
        base = (persistent, self.offloaded_frames) if self.offload_enabled else persistent
        if self.offload_enabled:
            if persistent is self._persistent_root:
                base = self._cache_root
            else:
                self._cache_ledger.add(base)
                self._cache_ledger.remove(id(self._cache_root))
        working = (base, self.frames, self.snapshots, self.counts, self.last_event, self._score_keys, self._score_queries)
        if all(a is b for a, b in zip(working, self._working_root)):
            working = self._working_root
        else:
            self._working_ledger.add(working)
            self._working_ledger.remove(id(self._working_root))
        self._persistent_root, self._working_root = persistent, working
        self._cache_root = base
        self._persistent_ledger.refresh_metadata_sizes()
        self._working_ledger.refresh_metadata_sizes()
        self._offload_ledger.refresh_metadata_sizes()
        if self.offload_enabled:
            self._cache_ledger.refresh_metadata_sizes()

    @property
    def recent_ids(self):
        key = (id(self.frames), self.frames.key_version, self.recent_frames)
        if getattr(self, '_recent_key', None) != key:
            self._recent_key = key
            self._recent_ids = tuple(sorted(self.frames)[-self.recent_frames:])
        return self._recent_ids

    def append_frame(self, timestamp, layers, *, score_keys=None, frame_queries=None, rope_start=0):
        """Append post-RoPE KV; supply pre-RoPE keys only when partitioning is needed."""
        if not math.isfinite(timestamp) or timestamp < self.last_timestamp:
            raise ValueError('Frame timestamps must be finite and nondecreasing')
        if len(layers) != len(self.sink):
            raise ValueError('Layer count mismatch')
        if any(k.device != self.device or v.device != self.device for k, v in layers):
            raise ValueError('Frame KV must already be on the cache device')
        records = tuple(LayerFrame.full(k, v, self.dtype) for k, v in layers)
        if any(x.high_k.shape != (self.sink[i][0].shape[0], self.frame_tokens,
                                  self.sink[i][0].shape[-1]) for i, x in enumerate(records)):
            raise ValueError('Frame KV shape mismatch')
        if not isinstance(rope_start, int) or rope_start < 0:
            raise ValueError('Frame RoPE start must be a nonnegative integer')
        if score_keys is not None:
            if len(score_keys) != len(records) or any(
                    key.device != self.device or key.shape != record.high_k.shape
                    for key, record in zip(score_keys, records)):
                raise ValueError('Pre-RoPE scoring keys must match frame KV shapes and device')
            score_keys = tuple(resident_copy(key, self.dtype) for key in score_keys)
        if frame_queries is not None:
            if len(frame_queries) != len(records) or any(
                    q is None or q.ndim != 2 or q.device != self.device or q.shape[0] < 1
                    or q.shape[0] % record.high_k.shape[0] or q.shape[-1] != record.high_k.shape[-1]
                    for q, record in zip(frame_queries, records)):
                raise ValueError('Frame Query means must match layer heads, dimensions and device')
            frame_queries = tuple(q.detach().float().clone() for q in frame_queries)
        frame = Frame(self.next_id, float(timestamp), self.segment_id, records,
                      rope_start=rope_start, token_count=self.frame_tokens)
        self.frames[frame.frame_id] = frame
        if score_keys is not None:
            self._score_keys[frame.frame_id] = score_keys
        if frame_queries is not None:
            self._score_queries[frame.frame_id] = frame_queries
        self.last_timestamp = timestamp
        self.next_id += 1
        self.version += 1
        return frame.frame_id

    def _persistent(self):
        self._refresh_accounting()
        return self._persistent_root

    def persistent_bytes(self):
        self._refresh_accounting()
        return self._persistent_ledger.total

    def _classify_segment(self, pending, queries, scoring_strategy='summary'):
        # Keep temporary scoring tensors scoped here so they are released before
        # budget enforcement, including when a later layer fails validation.
        frame_keys = [self._score_keys.get(frame.frame_id) for frame in pending]
        if any(keys is None for keys in frame_keys):
            raise ValueError('Missing pre-RoPE scoring keys')
        if scoring_strategy == 'frame_mean_query':
            frame_queries = [self._score_queries.get(frame.frame_id) for frame in pending]
            if any(row is None for row in frame_queries):
                raise ValueError('Missing frame Query means')
            queries = (torch.stack([row[layer] for row in frame_queries]) for layer in range(len(self.sink)))
        classified = [[] for _ in pending]
        for layer, query in enumerate(queries):
            keys = [row[layer] for row in frame_keys]
            if any(key.shape != frame.layers[layer].high_k.shape for frame, key in zip(pending, keys)):
                raise ValueError('Scoring key must match the unpartitioned frame')
            scorer = segment_frame_query_scores if scoring_strategy == 'frame_mean_query' else segment_summary_scores
            frame_local = scoring_strategy == 'frame_mean_query' and self.scoring_scope == 'frame'
            scorer = frame_local_query_scores if frame_local else scorer
            scores = scorer(query, keys, headwise=self.compression_granularity == 'head')
            entries = [frame.layers[layer] for frame in pending]
            if scoring_strategy == 'frame_mean_query' and self.compression_granularity == 'head':
                partitioned = LayerFrame.classify_batch(entries, scores)
            else:
                partitioned = [entry.classify_scores(row) for entry, row in zip(entries, scores)]
            for row, entry in zip(classified, partitioned):
                row.append(entry)
        return {frame.frame_id: tuple(row) for frame, row in zip(pending, classified)}

    def clear_frame_queries(self):
        """Release temporary frame means, including an interrupted segment."""
        self._score_queries.clear()

    def close_segment(self, queries, error='', *, scoring_strategy='summary'):
        if scoring_strategy not in ('summary', 'frame_mean_query'):
            raise ValueError('Invalid scoring_strategy')
        if self.scoring_scope == 'frame' and scoring_strategy != 'frame_mean_query':
            raise ValueError('scoring_scope=frame requires frame_mean_query scoring')
        began = time.monotonic()
        pending = [self.frames[i] for i in self._open_ids]
        if not pending:
            return self.memory_stats()
        with measure(self.profiler, 'token_scoring'):
            classified = {}
            try:
                if scoring_strategy == 'summary' and (queries is None or len(queries) != len(self.sink)):
                    raise ValueError(error or 'summary_empty')
                classified = self._classify_segment(pending, queries, scoring_strategy)
            except (ValueError, RuntimeError) as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
                error = (error or str(exc))[:256]
                classified = {f.frame_id: f.layers for f in pending}
                self.counts['scoring_failures'] += 1
            for frame in pending:
                self.frames[frame.frame_id] = replace(frame, layers=classified[frame.frame_id],
                                                     closed=True, score_error=error)
            # Drop local references before budgeting; snapshots explicitly own older versions.
            del pending, classified
        self.segment_id += 1
        self.version += 1
        self.last_event = dict(version=self.version, quantized=0, dropped_low=0, evicted=0,
                               offloaded=0,
                               before_bytes=0, after_bytes=0, elapsed_seconds=0.0,
                               trigger='frame_batch_commit' if self.scoring_scope == 'frame' else 'segment_commit',
                               score_error=error)
        self.last_event['before_bytes'] = self.persistent_bytes()
        self._enforce_budget()
        self.last_event['elapsed_seconds'] = time.monotonic()-began
        self.last_event['after_bytes'] = self.persistent_bytes()
        if self.last_event['after_bytes'] > self.budget:
            raise MemoryError('Protected cache exceeds budget')
        return self.memory_stats()

    def close_uncompressed_segment(self, *, return_stats=True):
        """Comparison baseline: retain every frame in model dtype; no policy calls."""
        pending = [self.frames[i] for i in self._open_ids]
        for frame in pending:
            self.frames[frame.frame_id] = replace(frame, closed=True)
        if pending:
            self.segment_id += 1
            self.version += 1
        return self.memory_stats() if return_stats else None

    def _enforce_budget(self):
        if self.persistent_bytes() <= self.budget:
            return
        self.counts['compressions'] += 1
        recent = set(self.recent_ids)
        candidates = sorted((f for f in self.frames.values() if f.closed and f.frame_id not in recent),
                            key=lambda f: (f.timestamp, f.frame_id))
        # Discard all eligible low regions before spending precision on high regions.
        with measure(self.profiler, 'token_pruning'):
            for old in candidates:
                frame = self.frames[old.frame_id]
                if not any(x.low_indices.numel() for x in frame.layers):
                    continue
                self.frames[frame.frame_id] = replace(frame, layers=tuple(x.drop_low() for x in frame.layers))
                self.counts['dropped_low_frames'] += 1
                self.last_event['dropped_low'] += 1
        if self.persistent_bytes() <= self.budget:
            return
        with measure(self.profiler, 'quantization'):
            for old in candidates:
                frame = self.frames[old.frame_id]
                layers = tuple(x.shrink(self.group_size) for x in frame.layers)
                self.last_event['quantized'] += sum(a is not b for a, b in zip(frame.layers, layers))
                self.frames[frame.frame_id] = replace(frame, layers=layers)
        if self.persistent_bytes() <= self.budget:
            return
        with measure(self.profiler, 'frame_eviction'):
            # LFU, then FIFO: frame_id is assigned monotonically on insertion.
            for old in sorted(candidates, key=lambda f: (f.hits, f.frame_id)):
                if self.offload_enabled:
                    frame = self.frames[old.frame_id]
                    self.offloaded_frames[frame.frame_id] = replace(frame, layers=tuple(
                        entry.to('cpu', pin_memory=self.device.type == 'cuda') for entry in frame.layers))
                del self.frames[old.frame_id]
                if self.offload_enabled:
                    self.counts['offloaded_frames'] += 1
                    self.last_event['offloaded'] = self.last_event.get('offloaded', 0) + 1
                else:
                    self.counts['evicted_frames'] += 1
                    self.last_event['evicted'] += 1
                if self.persistent_bytes() <= self.budget:
                    return
        if self.persistent_bytes() > self.budget:
            raise MemoryError('Budget cannot hold the protected cache')

    def acquire_snapshot(self, question_id, cutoff=None):
        if self.snapshots or question_id <= self.last_snapshot_id:
            raise ValueError('Only one snapshot is allowed; question IDs must increase')
        if cutoff is not None and self.last_timestamp > cutoff:
            raise ValueError('Historical snapshots unavailable: submit questions in stream order')
        frames = tuple(sorted((*self.frames.values(), *self.offloaded_frames.values()), key=lambda f: f.frame_id))
        snapshot = CacheSnapshot(self.version, question_id, self.sink, frames, self.recent_ids)
        self.snapshots[question_id] = snapshot
        self.last_snapshot_id = question_id
        return snapshot

    def release_snapshot(self, question_id):
        self.snapshots.pop(question_id, None)

    def record_activation(self, question_id, frame_ids):
        if question_id <= self.last_activation_id:
            return
        if question_id > self.last_snapshot_id:
            raise ValueError('Activation for an unknown question')
        for frame_id in set(frame_ids):
            table = self.frames if frame_id in self.frames else self.offloaded_frames
            if frame_id in table:
                frame = table[frame_id]
                table[frame_id] = replace(frame, hits=frame.hits+1)
        self.last_activation_id = question_id

    def frame_states(self, segment_ids=None):
        """Detached diagnostics; counts are mean token positions per KV head.

        Each row holds [unquantized, INT8, INT4, INT2] counts for every decoder layer.
        Missing frame IDs below next_id were evicted; future frames are not emitted.
        No tensors, snapshots or GPU-to-CPU payload copies are retained by this report.
        """
        recent = set(self.recent_ids)
        result = []
        for frame_id in range(self.next_id):
            offloaded = frame_id in self.offloaded_frames
            frame = self.offloaded_frames[frame_id] if offloaded else self.frames.get(frame_id)
            if frame is None:
                result.append(dict(frame_id=frame_id, segment_id=(segment_ids or {}).get(frame_id),
                                   status='evicted', protected=False, counts=None, head_counts=None))
                continue
            counts = [list(row) for row in (self._token_counts(frame) if offloaded else self._frame_counts[frame_id])]
            head_counts = []
            for layer in frame.layers:
                rows = []
                for h in range(layer.heads):
                    low = (layer.low_indices.numel() if layer.low_offsets is None else
                           layer.low_offsets[h+1]-layer.low_offsets[h])
                    high = (layer.high_indices.numel() if layer.high_offsets is None else
                            layer.high_offsets[h+1]-layer.high_offsets[h])
                    row = [low, 0, 0, 0]
                    row[{16: 0, 8: 1, 4: 2, 2: 3}[layer.bits]] += high
                    rows.append(row)
                head_counts.append(rows)
            result.append(dict(frame_id=frame_id, segment_id=frame.segment_id,
                               status='offloaded' if offloaded else 'closed' if frame.closed else 'open',
                               protected=frame_id in recent, counts=counts, head_counts=head_counts))
        return dict(layer_count=len(self.sink), tokens_per_frame=self.frame_tokens,
                    kv_heads=[k.shape[0] for k, _ in self.sink], count_unit='mean_tokens_per_kv_head',
                    bit_order=[16, 8, 4, 2], full_precision_dtype=str(self.dtype).removeprefix('torch.'), frames=result)

    def memory_stats(self):
        self._refresh_accounting()
        ledger = self._persistent_ledger
        pbytes = ledger.total
        working = self._working_ledger.total-self._cache_ledger.total
        return dict(cache_device=str(self.device), cache_dtype=str(self.dtype).removeprefix('torch.'), offload_enabled=self.offload_enabled,
                    compression_granularity=self.compression_granularity,
                    scoring_scope=self.scoring_scope,
                    budget_bytes=self.budget, persistent_bytes=pbytes,
                    persistent_tensor_bytes=ledger.tensor_bytes,
                    persistent_cuda_bytes=ledger.cuda_bytes,
                    cache_working_bytes=working, retained_frames=len(self.frames)+len(self.offloaded_frames),
                    resident_frames=len(self.frames), offloaded_frames=len(self.offloaded_frames),
                    offloaded_bytes=self._offload_ledger.total if self.offload_enabled else 0,
                    offloaded_tensor_bytes=self._offload_ledger.tensor_bytes,
                    open_frames=len(self._open_ids),
                    bits=dict(self._bits), counts=dict(self.counts), last_event=dict(self.last_event))

    def audit_accounting(self):
        """Slow independent verification for tests/debugging, never budget checks."""
        stats = self.memory_stats()
        assert self.recent_ids == tuple(sorted(self.frames)[-self.recent_frames:])
        recent = set(self.recent_ids)
        expected_frames = {i: f for i, f in self.frames.items() if f.closed or i in recent}
        assert self._managed.keys() == expected_frames.keys()
        assert all(self._managed[i] is f for i, f in expected_frames.items())
        assert set(self._open_ids) == {i for i, f in self.frames.items() if not f.closed}
        persistent = self._persistent_root
        expected = dict(persistent_bytes=footprint(persistent),
                        persistent_tensor_bytes=tensor_storage_bytes(persistent),
                        persistent_cuda_bytes=tensor_storage_bytes(persistent, 'cuda'),
                        cache_working_bytes=footprint(self._working_root)-footprint(self._cache_root),
                        offloaded_bytes=footprint(self.offloaded_frames) if self.offload_enabled else 0,
                        offloaded_tensor_bytes=tensor_storage_bytes(self.offloaded_frames))
        assert not (self.frames.keys() & self.offloaded_frames.keys())
        assert self._offload_ledger.cuda_bytes == 0
        for name, value in expected.items():
            if stats[name] != value:
                raise AssertionError(f'{name}: ledger={stats[name]}, full scan={value}')
        return stats
