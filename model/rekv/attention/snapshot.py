"""Zero-copy, published video prefixes and private per-question retrieval contexts.

There is one writer. It only appends history/representatives, and replaces
init/remainder tensors instead of modifying their published storage. Readers
retain the old storage and a fixed prefix length until their CUDA work ends.
"""
from collections.abc import Sequence
from dataclasses import dataclass, field
from operator import index as as_index
from threading import Lock

import torch

from .kv_trace import TRACE
from .retrieval import select_blocks


@dataclass(frozen=True)
class CPUBlockPrefix(Sequence):
    """Bounded read access to an append-only history, without copying its index.

    Only cpu_data is exposed; MemoryUnit's mutable GPU residency is private to
    the writer. The retained list must never have existing entries replaced or
    removed. Reset creates a new list, so an older view remains valid.
    """
    _history: list
    length: int

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if isinstance(item, slice):
            return tuple(self[i] for i in range(*item.indices(self.length)))
        item = as_index(item)
        if item < 0:
            item += self.length
        if not 0 <= item < self.length:
            raise IndexError('Block is outside the published prefix')
        return self._history[item].cpu_data


@dataclass(frozen=True)
class LayerSnapshot:
    n_init: int
    n_local: int
    topk: int
    chunk_size: int
    block_size: int
    init_exc: bool
    init_kv: tuple
    remainder: tuple
    block_keys: tuple
    blocks: tuple
    num_global_block: int
    length: int

    @classmethod
    def capture(cls, cache):
        # Publish between complete frames. The enclosing KVSnapshot records the
        # ready event after these waits, including asynchronous CPU offloads.
        if cache.local_k.is_cuda and cache.global_stream is not None:
            torch.cuda.current_stream(cache.local_k.device).wait_stream(cache.global_stream)
        return cls(cache.n_init, cache.n_local, cache.topk, cache.chunk_size,
                   cache.block_size, cache.init_exc,
                   (cache.init_k, cache.init_v),
                   cache.global_remainder,
                   tuple(v.get_data() for v in cache.block_k),
                   tuple(CPUBlockPrefix(row, cache.num_global_block) for row in cache.global_blocks),
                   cache.num_global_block, cache.length)

    def select(self, query):
        return select_blocks(query, n_init=self.n_init, topk=self.topk,
                             chunk_size=self.chunk_size, block_size=self.block_size,
                             init_exc=self.init_exc, remainder_k=self.remainder[0],
                             block_keys=self.block_keys, num_global_block=self.num_global_block)[0]

    def retrieve(self, indices):
        source = self.init_kv if self.init_exc else self.remainder
        batch, heads, _, dim = source[0].shape
        available = self.num_global_block if self.init_exc else (
            self.remainder[0].size(2) - self.n_init) // self.block_size
        if len(indices) != batch or len({len(row) for row in indices}) != 1:
            raise ValueError('Retrieved indices must have equal lengths for each batch item')
        if any(i < 0 or i >= available for row in indices for i in row):
            raise ValueError('Retrieved index is outside the snapshot prefix')
        count = len(indices[0])
        if count > self.topk or count * self.block_size > self.n_local:
            raise ValueError('Retrieved KV exceeds the configured retrieval/local window')
        length = self.n_init + count * self.block_size
        result = tuple(t.new_empty((batch, heads, length, dim)) for t in source)
        for out, initial in zip(result, source):
            out[:, :, :self.n_init].copy_(initial[:, :, :self.n_init], non_blocking=True)
        for u, row in enumerate(indices):
            for offset, index in enumerate(row):
                start = self.n_init + offset * self.block_size
                if self.init_exc:
                    payload = self.blocks[u][index]
                else:
                    src = self.n_init + index * self.block_size
                    payload = tuple(t[u, :, src:src + self.block_size] for t in self.remainder)
                for out, value in zip(result, payload):
                    out[u, :, start:start + self.block_size].copy_(value, non_blocking=True)
                if self.init_exc and TRACE is not None:
                    TRACE.record('load_h2d', payload)
        return result

    def record_stream(self, stream):
        for tensor in (*self.init_kv, *self.remainder, *self.block_keys):
            tensor.record_stream(stream)


@dataclass
class SnapshotRetrieval:
    """One forward's state; never attached to the live encoding cache."""
    snapshot: LayerSnapshot
    indices: object = None

    def get_retrieved_kv(self, query):
        if self.indices is None:
            self.indices = self.snapshot.select(query)
        elif isinstance(self.indices, torch.Tensor):
            self.indices = self.indices.cpu().tolist()
        return self.snapshot.retrieve(self.indices)


@dataclass(frozen=True)
class KVSnapshot:
    layers: tuple
    ready: object
    position_bias: object
    _consumer_streams: set = field(default_factory=set, init=False, repr=False, compare=False)
    _consumer_lock: object = field(default_factory=Lock, init=False, repr=False, compare=False)

    def wait(self):
        stream = torch.cuda.current_stream(self.layers[0].remainder[0].device)
        # Stream ordering covers all subsequent questions using this view on
        # the same stream. Retain stream objects as well as the tensor storage.
        with self._consumer_lock:
            if stream not in self._consumer_streams:
                stream.wait_event(self.ready)
                for layer in self.layers:
                    layer.record_stream(stream)
                self._consumer_streams.add(stream)

    def retrieval_contexts(self, indices=None):
        return tuple(SnapshotRetrieval(layer, indices) for layer in self.layers)
