"""Reusable materializations of the resident, compressed video prefix."""
import torch

from .attention import EncodeLayer, RotaryCache
from .packed_attention import HeadwiseRotaryCache


class EncodingPrefix:
    def __init__(self):
        self.entries = {}
        self.working_bytes = 0

    def clear(self):
        self.entries.clear()
        self.working_bytes = 0

    def prune(self, frames):
        # Compare immutable layer payloads, not Frame identities: QA can update
        # hit counts without changing KV. Removed/offloaded frames are excluded.
        for key, (entry, parts, size) in tuple(self.entries.items()):
            frame_id, layer = key
            frame = frames.get(frame_id)
            if frame is None or frame.layers[layer] is not entry:
                del self.entries[key]
                self.working_bytes -= size

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
        if cached is not None:
            self.working_bytes -= cached[2]
        self.entries[key] = (entry, parts, size)
        self.working_bytes += size
        return parts

    def contexts(self, cache, *, capture_scores, capture_frame_query):
        self.prune(cache.frames)
        frames = sorted(cache.frames.values(), key=lambda f: f.frame_id)
        contexts = []
        for layer, (sink_k, sink_v) in enumerate(cache.sink):
            materialized = [self._materialize(frame, layer) for frame in frames]
            sink_positions = torch.arange(sink_k.shape[1], device=sink_k.device)
            headwise = any(frame.layers[layer].high_offsets is not None for frame in frames)

            def assemble(head=None):
                parts = []
                for frame, row in zip(frames, materialized):
                    if frame.layers[layer].high_offsets is None:
                        k, v, ids = row[0]
                        if head is not None:
                            k, v = k[head:head+1], v[head:head+1]
                    else:
                        k, v, ids = row[head]
                    parts.append((k, v, ids.long() + frame.rope_start))
                sk = sink_k if head is None else sink_k[head:head+1]
                sv = sink_v if head is None else sink_v[head:head+1]
                return RotaryCache(torch.cat([sk] + [p[0] for p in parts], 1)[None],
                                   torch.cat([sv] + [p[1] for p in parts], 1)[None],
                                   torch.cat([sink_positions] + [p[2] for p in parts]))

            past = (HeadwiseRotaryCache(tuple(assemble(h) for h in range(sink_k.shape[0])))
                    if headwise else assemble())
            contexts.append(EncodeLayer(past, capture_scores=capture_scores,
                capture_frame_query=capture_frame_query,
                next_position=sink_k.shape[1] + cache.next_id * cache.frame_tokens))
        return contexts
