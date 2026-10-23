"""Single-writer immutable frame cache. KV stays on its originating device; snapshots never leave the owner process."""
from dataclasses import dataclass, is_dataclass, replace
import sys
import time

import torch

from .quantization import PackedKV
from .segments import summary_scores


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


def resident_copy(x):
    return x.detach().to(dtype=torch.float16).clone().contiguous()


@dataclass(frozen=True)
class LayerFrame:
    high_indices: torch.Tensor
    high_k: torch.Tensor
    high_v: torch.Tensor
    low_indices: torch.Tensor
    low_k: object
    low_v: object
    representative: torch.Tensor
    bits: int = 16

    @classmethod
    def full(cls, key, value):
        key, value = resident_copy(key), resident_copy(value)
        if key.shape != value.shape or key.ndim != 3:
            raise ValueError('Expected matching [kv_heads, tokens, head_dim] tensors')
        return cls(torch.arange(key.shape[1], dtype=torch.int32, device=key.device), key, value,
                   torch.empty(0, dtype=torch.int32, device=key.device), None, None,
                   key.float().mean(1).half())

    def classify(self, query):
        scores = summary_scores(query, self.high_k)
        high = scores >= scores.mean()
        return replace(self, high_indices=self.high_indices[high].clone(),
                       high_k=self.high_k[:, high].clone(), high_v=self.high_v[:, high].clone(),
                       low_indices=self.high_indices[~high].clone(),
                       low_k=self.high_k[:, ~high].clone(), low_v=self.high_v[:, ~high].clone())

    def shrink(self, group_size):
        if not self.low_indices.numel() or self.bits == 2:
            return self
        if self.bits == 16:
            key = PackedKV.quantize(self.low_k, group_size).shrink()
            value = PackedKV.quantize(self.low_v, group_size).shrink()
        else:
            key, value = self.low_k.shrink(), self.low_v.shrink()
        return replace(self, low_k=key, low_v=value, bits=key.bits)

    def drop_low(self):
        return replace(self, low_indices=torch.empty(0, dtype=torch.int32, device=self.high_k.device),
                       low_k=None, low_v=None, bits=0)

    def materialize(self, device=None):
        device = self.high_k.device if device is None else torch.device(device)
        if device != self.high_k.device:
            raise ValueError('Cache materialization must stay on the cache device; offload is disabled')
        keys, values = [self.high_k.to(device)], [self.high_v.to(device)]
        indices = [self.high_indices]
        if self.low_indices.numel():
            keys.append(self.low_k.dequantize(device) if isinstance(self.low_k, PackedKV) else self.low_k.to(device))
            values.append(self.low_v.dequantize(device) if isinstance(self.low_v, PackedKV) else self.low_v.to(device))
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
    hits: int = 0
    score_error: str = ''


@dataclass(frozen=True)
class CacheSnapshot:
    version: int
    question_id: int
    sink: tuple
    frames: tuple
    recent_ids: tuple

    def retrieve_layer(self, layer, query, topk=64):
        """Select stable frame IDs, then materialize a ragged per-layer sequence."""
        if topk < len(self.recent_ids):
            raise ValueError('topk must accommodate the recent frames')
        protected = set(self.recent_ids)
        available = [f for f in self.frames if f.frame_id not in protected]
        if query.device != self.sink[layer][0].device:
            raise ValueError('Query and resident cache must be on the same device')
        q = query.detach().float().mean(dim=-2).squeeze(0)
        if available:
            keys = torch.stack([f.layers[layer].representative for f in available]).float()
            if q.shape[0] % keys.shape[1]:
                raise ValueError('Invalid GQA heads')
            keys = keys.repeat_interleave(q.shape[0] // keys.shape[1], 1)
            scores = (keys*q).sum((1, 2))
            ranking = scores.argsort(descending=True, stable=True).tolist()
            available = [available[i] for i in ranking]
        selected = [f for f in self.frames if f.frame_id in protected] + available[:topk-len(protected)]
        selected.sort(key=lambda f: (f.timestamp, f.frame_id))
        device = query.device
        keys, values = [self.sink[layer][0].to(device)], [self.sink[layer][1].to(device)]
        positions = []
        for frame in selected:
            key, value, ids = frame.layers[layer].materialize(device)
            keys.append(key)
            values.append(value)
            positions.append((frame.frame_id, ids))
        return (torch.cat(keys, 1)[None], torch.cat(values, 1)[None],
                tuple(f.frame_id for f in selected), positions)


class StreamingKVCache:
    def __init__(self, sink, budget_tokens, frame_tokens, recent_frames=2, group_size=128):
        if budget_tokens <= 0 or recent_frames < 1 or frame_tokens < 1:
            raise ValueError('Budget, recent_frames and frame_tokens must be positive')
        if group_size < 1 or group_size % 8:
            raise ValueError('group_size must be a positive multiple of eight')
        self.sink = tuple((resident_copy(k), resident_copy(v)) for k, v in sink)
        if not self.sink:
            raise ValueError('At least one layer is required')
        self.device = self.sink[0][0].device
        if any(k.device != self.device or v.device != self.device for k, v in self.sink):
            raise ValueError('All cache layers must use one device')
        self.budget = budget_tokens * 4 * sum(k.shape[0]*k.shape[-1] for k, _ in self.sink)
        self.frame_tokens, self.recent_frames, self.group_size = frame_tokens, recent_frames, group_size
        self.frames = {}
        self.next_id = self.segment_id = self.version = 0
        self.last_timestamp = float('-inf')
        self.last_activation_id = -1
        self.last_snapshot_id = -1
        self.snapshots = {}
        self.counts = dict(compressions=0, dropped_low_frames=0, evicted_frames=0, scoring_failures=0)
        self.last_event = {}
        # Conservative protected-region reserve, including split-frame tensor metadata.
        layers = tuple(LayerFrame.full(torch.zeros(k.shape[0], frame_tokens, k.shape[-1], dtype=torch.float16),
                                      torch.zeros(k.shape[0], frame_tokens, k.shape[-1], dtype=torch.float16))
                       for k, _ in self.sink)
        reserve = tuple(Frame(i, float(i), 0, tuple(replace(x,
            low_k=torch.empty(x.high_k.shape[0], 0, x.high_k.shape[-1], dtype=torch.float16),
            low_v=torch.empty(x.high_v.shape[0], 0, x.high_v.shape[-1], dtype=torch.float16))
            for x in layers), True) for i in range(recent_frames))
        # Each recent frame owns separate storage; the prototype above shares it.
        protected = footprint((self.sink, reserve)) + (recent_frames-1)*footprint(layers) + 4096
        if self.budget < protected:
            raise ValueError(f'Budget {self.budget} bytes cannot hold sink/recent metadata; need at least {protected}')

    @property
    def recent_ids(self):
        return tuple(sorted(self.frames)[-self.recent_frames:])

    def append_frame(self, timestamp, layers):
        if not torch.isfinite(torch.tensor(timestamp)) or timestamp < self.last_timestamp:
            raise ValueError('Frame timestamps must be finite and nondecreasing')
        if len(layers) != len(self.sink):
            raise ValueError('Layer count mismatch')
        if any(k.device != self.device or v.device != self.device for k, v in layers):
            raise ValueError('Frame KV must already be on the cache device')
        records = tuple(LayerFrame.full(k, v) for k, v in layers)
        if any(x.high_k.shape != (self.sink[i][0].shape[0], self.frame_tokens,
                                  self.sink[i][0].shape[-1]) for i, x in enumerate(records)):
            raise ValueError('Frame KV shape mismatch')
        frame = Frame(self.next_id, float(timestamp), self.segment_id, records)
        self.frames[frame.frame_id] = frame
        self.last_timestamp = timestamp
        self.next_id += 1
        self.version += 1
        return frame.frame_id

    def _persistent(self):
        recent = set(self.recent_ids)
        return (self.sink, {i: f for i, f in self.frames.items() if f.closed or i in recent},
                self.budget, self.frame_tokens, self.recent_frames, self.group_size,
                self.next_id, self.segment_id, self.version, self.last_activation_id)

    def persistent_bytes(self):
        return footprint(self._persistent())

    def close_segment(self, queries, error=''):
        began = time.monotonic()
        pending = [f for f in self.frames.values() if not f.closed]
        if not pending:
            return self.memory_stats()
        classified = {}
        try:
            if queries is None or len(queries) != len(self.sink):
                raise ValueError(error or 'summary_empty')
            for frame in pending:
                classified[frame.frame_id] = tuple(x.classify(q) for x, q in zip(frame.layers, queries))
        except (ValueError, RuntimeError) as exc:
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
                               before_bytes=0, after_bytes=0, elapsed_seconds=0.0,
                               trigger='segment_commit', score_error=error)
        self.last_event['before_bytes'] = self.persistent_bytes()
        self._enforce_budget()
        self.last_event['elapsed_seconds'] = time.monotonic()-began
        self.last_event['after_bytes'] = self.persistent_bytes()
        if self.persistent_bytes() > self.budget:
            raise MemoryError('Protected cache exceeds budget')
        return self.memory_stats()

    def _enforce_budget(self):
        if self.persistent_bytes() <= self.budget:
            return
        self.counts['compressions'] += 1
        recent = set(self.recent_ids)
        candidates = sorted((f for f in self.frames.values() if f.closed and f.frame_id not in recent),
                            key=lambda f: (f.timestamp, f.frame_id))
        for frame in candidates:
            layers = tuple(x.shrink(self.group_size) for x in frame.layers)
            self.last_event['quantized'] += sum(a is not b for a, b in zip(frame.layers, layers))
            self.frames[frame.frame_id] = replace(frame, layers=layers)
        if self.persistent_bytes() <= self.budget:
            return
        for old in candidates:
            frame = self.frames[old.frame_id]
            if not any(x.low_indices.numel() for x in frame.layers):
                continue
            self.frames[frame.frame_id] = replace(frame, layers=tuple(x.drop_low() for x in frame.layers))
            self.counts['dropped_low_frames'] += 1
            self.last_event['dropped_low'] += 1
            if self.persistent_bytes() <= self.budget:
                return
        for old in sorted(candidates, key=lambda f: (f.hits, f.timestamp, f.frame_id)):
            del self.frames[old.frame_id]
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
        frames = tuple(self.frames.values())
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
            if frame_id in self.frames:
                frame = self.frames[frame_id]
                self.frames[frame_id] = replace(frame, hits=frame.hits+1)
        self.last_activation_id = question_id

    def memory_stats(self):
        persistent = self._persistent()
        pbytes = footprint(persistent)
        working = footprint((persistent, self.frames, self.snapshots, self.counts, self.last_event)) - pbytes
        bits = {str(b): 0 for b in (16, 8, 4, 2, 0)}
        for frame in self.frames.values():
            for layer in frame.layers:
                bits[str(layer.bits)] += 1
        return dict(cache_device=str(self.device), offload_enabled=False,
                    budget_bytes=self.budget, persistent_bytes=pbytes,
                    cache_working_bytes=working, retained_frames=len(self.frames),
                    open_frames=sum(not f.closed for f in self.frames.values()),
                    bits=bits, counts=dict(self.counts), last_event=dict(self.last_event))
