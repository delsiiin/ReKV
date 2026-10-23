"""Incremental accounting for the cache's immutable object graphs.

Nodes contain identities, sizes and edges, never strong references to tensors.
Reference counts deduplicate Python objects and underlying tensor storages
independently. Replacing a frame visits only newly owned subtrees; unchanged
layers and tensors are shared with the old record and snapshots.
"""
from dataclasses import dataclass, is_dataclass
import sys
import weakref

import torch


@dataclass(slots=True)
class _Node:
    size: int
    refs: int = 1
    children: object = ()
    storage: object = None
    owner_type: object = None


class MemoryLedger:
    def __init__(self):
        self.nodes = {}
        self.storages = {}
        self.total = self.tensor_bytes = self.cuda_bytes = 0
        self._class_dicts = {}

    def add(self, value):
        identity = id(value)
        node = self.nodes.get(identity)
        if node is not None:
            node.refs += 1
            return identity
        node = _Node(sys.getsizeof(value))
        self.nodes[identity] = node
        self.total += node.size
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            node.storage = key
            refs = self.storages.get(key, 0)
            self.storages[key] = refs+1
            if not refs:
                self.total += key[2]
                self.tensor_bytes += key[2]
                if value.device.type == 'cuda':
                    self.cuda_bytes += key[2]
        elif is_dataclass(value):
            attributes = vars(value)
            child = self.add(attributes)
            node.children = (child,)
            self.nodes[child].owner_type = type(value)
            self._class_dicts.setdefault(type(value), {})[child] = weakref.ref(value)
        elif isinstance(value, dict):
            node.children = {k: (self.add(k), self.add(v)) for k, v in value.items()}
        elif isinstance(value, (tuple, list, set)):
            node.children = tuple(self.add(v) for v in value)
        return identity

    def remove(self, identity):
        node = self.nodes[identity]
        node.refs -= 1
        if node.refs:
            return
        del self.nodes[identity]
        self.total -= node.size
        if node.storage is not None:
            key = node.storage
            refs = self.storages[key]-1
            if refs:
                self.storages[key] = refs
            else:
                del self.storages[key]
                self.total -= key[2]
                self.tensor_bytes -= key[2]
                if key[0].split(':')[0] == 'cuda':
                    self.cuda_bytes -= key[2]
        if node.owner_type is not None:
            members = self._class_dicts[node.owner_type]
            del members[identity]
            if not members:
                del self._class_dicts[node.owner_type]
        if isinstance(node.children, dict):
            for key, value in node.children.values():
                self.remove(key)
                self.remove(value)
        else:
            for child in node.children:
                self.remove(child)

    def resize(self, value):
        node = self.nodes.get(id(value))
        if node is not None:
            size = sys.getsizeof(value)
            self.total += size-node.size
            node.size = size

    def mapping_changed(self, mapping, key):
        """Refresh one edge after insertion, replacement or deletion."""
        node = self.nodes.get(id(mapping))
        if node is None:
            return
        previous = node.children.get(key)
        if key in mapping:
            value = mapping[key]
            if previous is None:
                node.children[key] = (self.add(key), self.add(value))
            elif previous[1] != id(value):
                # Register shared descendants before releasing the old version.
                node.children[key] = (previous[0], self.add(value))
                self.remove(previous[1])
        elif previous is not None:
            del node.children[key]
            self.remove(previous[0])
            self.remove(previous[1])
        self.resize(mapping)

    def refresh_metadata_sizes(self):
        # CPython shared-key instance dictionaries shrink while the first few
        # instances of a class are created, including instances outside this
        # ledger. Probe one live instance per class; revisit its dictionaries
        # only when that class's allocation size changes (bounded warmup).
        for members in self._class_dicts.values():
            for identity, ref in members.items():
                value = ref()
                if value is not None:
                    size = sys.getsizeof(vars(value))
                    if size != self.nodes[identity].size:
                        for child, other in members.items():
                            instance = other()
                            if instance is not None:
                                self.resize(vars(instance))
                    break


_MISSING = object()


class TrackedDict(dict):
    """dict-compatible mutation notifications, including public cache edits."""
    def __init__(self, values=(), *, changed):
        super().__init__(values)
        # A bound callback must not create cache -> table -> cache cycles: reset
        # should release GPU tensors immediately, without waiting for Python GC.
        self._changed = weakref.WeakMethod(changed)
        self.key_version = 0

    def _notify(self, key, previous, value):
        changed = self._changed()
        if changed is not None:
            changed(self, key, previous, value)

    def __setitem__(self, key, value):
        previous = self.get(key, _MISSING)
        if previous is value:
            return
        super().__setitem__(key, value)
        if previous is _MISSING:
            self.key_version += 1
        self._notify(key, None if previous is _MISSING else previous, value)

    def __delitem__(self, key):
        previous = self[key]
        super().__delitem__(key)
        self.key_version += 1
        self._notify(key, previous, None)

    def pop(self, key, default=_MISSING):
        if key not in self:
            if default is _MISSING:
                raise KeyError(key)
            return default
        value = self[key]
        del self[key]
        return value

    def popitem(self):
        if not self:
            raise KeyError('popitem(): dictionary is empty')
        key = next(reversed(self))
        return key, self.pop(key)

    def clear(self):
        for key in list(self):
            del self[key]

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def __ior__(self, other):
        self.update(other)
        return self
