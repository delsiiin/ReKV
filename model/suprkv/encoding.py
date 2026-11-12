"""Reusable materializations of the resident, compressed video prefix."""
import torch
import weakref

from .attention import EncodeLayer, EncodingKV
from .packed_attention import HeadwiseRotaryCache
from .encoding_attention import EncodingSegments


class EncodingPrefix:
    def __init__(self):
        self.entries = {}
        self.working_bytes = 0
        self._by_frame = {}
        self._dirty = set()
        self._frames = None

    def clear(self):
        frames = self._frames() if self._frames is not None else None
        if frames is not None:
            frames.unwatch(self._changed)
        self._frames = None
        self._dirty.clear()
        self._by_frame.clear()
        self.entries.clear()
        self.working_bytes = 0

    def _changed(self, mapping, key, previous, value):
        if previous is None or value is None or previous.layers is not value.layers:
            self._dirty.add(key)

    def prune(self, frames):
        # Compare immutable layer payloads, not Frame identities: QA can update
        # hit counts without changing KV. Removed/offloaded frames are excluded.
        bound = self._frames() if self._frames is not None else None
        if not hasattr(frames, 'watch'):
            if bound is not None:
                bound.unwatch(self._changed)
            self._frames = None
            self._dirty.clear()
            dirty = tuple(self._by_frame)
        elif bound is not frames:
            if bound is not None:
                bound.unwatch(self._changed)
            self._frames = weakref.ref(frames)
            frames.watch(self._changed)
            dirty = tuple(self._by_frame)
            self._dirty.clear()
        else:
            dirty, self._dirty = self._dirty, set()
        for frame_id in dirty:
            frame = frames.get(frame_id)
            for key in tuple(self._by_frame.get(frame_id, ())):
                entry, parts, size = self.entries[key]
                if frame is None or frame.layers[key[1]] is not entry:
                    del self.entries[key]
                    self._by_frame[frame_id].remove(key)
                    self.working_bytes -= size
            if frame_id in self._by_frame and not self._by_frame[frame_id]:
                del self._by_frame[frame_id]

    def _materialize(self, frame, layer):
        key = (frame.frame_id, layer)
        entry = frame.layers[layer]
        cached = self.entries.get(key)
        if cached is not None and cached[0] is entry:
            return cached[1]
        if entry.high_offsets is None:
            if entry.bits == 16 and not entry.low_indices.numel():
                # Full frames already own suitable dense tensors. Do not copy.
                parts = ((entry.high_k, entry.high_v, entry.high_indices),)
                size = 0
            else:
                parts = (entry.materialize(),)
                size = sum(t.untyped_storage().nbytes() for t in parts[0])
        else:
            parts = tuple(entry.materialize_head(h) for h in range(entry.heads))
            size = sum(t.untyped_storage().nbytes() for part in parts for t in part)
        self._install(frame, layer, parts, size)
        return parts

    def _install(self, frame, layer, parts, size):
        key = (frame.frame_id, layer)
        entry = frame.layers[layer]
        cached = self.entries.get(key)
        if cached is not None:
            self.working_bytes -= cached[2]
        self.entries[key] = (entry, parts, size)
        self._by_frame.setdefault(frame.frame_id, set()).add(key)
        self.working_bytes += size

    def _prepare(self, frames):
        from .prefix_batch import supported, materialize_many
        pending = []
        for frame in frames:
            if len(self._by_frame.get(frame.frame_id, ())) == len(frame.layers):
                continue
            for layer, entry in enumerate(frame.layers):
                if (frame.frame_id, layer) in self.entries:
                    continue
                if (supported(entry) and not (entry.high_offsets is None and entry.bits == 16
                                              and not entry.low_indices.numel())):
                    pending.append((frame, layer))
                else:
                    self._materialize(frame, layer)
        if pending:
            restored = materialize_many([f.layers[l] for f, l in pending])
            for (frame, layer), parts in zip(pending, restored):
                size = sum(t.untyped_storage().nbytes() for part in parts for t in part)
                self._install(frame, layer, parts, size)

    def contexts(self, cache, *, capture_scores, capture_frame_query):
        self.prune(cache.frames)
        frames = sorted(cache.frames.values(), key=lambda f: f.frame_id)
        self._prepare(frames)
        contexts = []
        for layer, (sink_k, sink_v) in enumerate(cache.sink):
            materialized = [self._materialize(frame, layer) for frame in frames]
            headwise = any(frame.layers[layer].high_offsets is not None for frame in frames)

            def assemble(head=None):
                parts = []
                for frame, row in zip(frames, materialized):
                    if frame.layers[layer].high_offsets is None:
                        k, v, _ = row[0]
                        if head is not None:
                            k, v = k[head:head+1], v[head:head+1]
                    else:
                        k, v, _ = row[head]
                    parts.append((k, v))
                sk = sink_k if head is None else sink_k[head:head+1]
                sv = sink_v if head is None else sink_v[head:head+1]
                if head is not None and sink_k.is_cuda and sink_k.dtype in (torch.float16, torch.bfloat16):
                    return EncodingSegments(tuple([(sk, sv)]+parts))
                return EncodingKV(torch.cat([sk] + [p[0] for p in parts], 1)[None],
                                  torch.cat([sv] + [p[1] for p in parts], 1)[None])

            past = (HeadwiseRotaryCache(tuple(assemble(h) for h in range(sink_k.shape[0])))
                    if headwise else assemble())
            contexts.append(EncodeLayer(past, capture_scores=capture_scores,
                capture_frame_query=capture_frame_query,
                next_position=sink_k.shape[1] + cache.next_id * cache.frame_tokens))
        return contexts
