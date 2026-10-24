"""Incremental bytes must match an independent recursive scan at every step."""
from dataclasses import dataclass, replace
import gc
import unittest
import weakref
from unittest.mock import patch

import torch

from model.suprkv.cache import StreamingKVCache, footprint
from model.suprkv.accounting import MemoryLedger
from model.suprkv.quantization import PackedKV


def make_cache(device='cpu', frames=0, layers=2):
    sink = [(torch.zeros(2, 2, 16, device=device), torch.zeros(2, 2, 16, device=device)) for _ in range(layers)]
    cache = StreamingKVCache(sink, 100000, 8, group_size=8)
    for i in range(frames):
        append(cache, i)
    return cache


def append(cache, i):
    x = torch.arange(8, device=cache.device).view(1, 8, 1).expand(2, 8, 16).half()
    cache.append_frame(float(i), [(x, x)]*len(cache.sink), score_keys=[x]*len(cache.sink))


class AccountingTests(unittest.TestCase):
    def check(self, cache):
        stats = cache.audit_accounting()
        expected = {str(b): 0 for b in (16, 8, 4, 2, 0)}
        for frame in cache.frames.values():
            for layer in frame.layers:
                expected[str(layer.bits)] += 1
        self.assertEqual(stats['bits'], expected)
        # Cross-check cached token diagnostics against the actual high-region payload.
        rows = {row['frame_id']: row for row in cache.frame_states()['frames']}
        for frame in cache.frames.values():
            expected_counts = []
            for layer in frame.layers:
                bits = layer.high_k.bits if isinstance(layer.high_k, PackedKV) else 16
                counts = [layer.low_indices.numel(), 0, 0, 0]
                counts[{16: 0, 8: 1, 4: 2, 2: 3}[bits]] += layer.high_indices.numel()
                expected_counts.append(counts)
            self.assertEqual(rows[frame.frame_id]['counts'], expected_counts)
        return stats

    def test_python_instance_dictionary_warmup_is_accounted(self):
        @dataclass(frozen=True)
        class Record:
            value: object
            label: str
        records = (Record(torch.ones(4), 'first'), Record(torch.ones(4), 'second'))
        ledger = MemoryLedger()
        ledger.add(records)
        # Creating unrelated instances shrinks existing CPython split dictionaries.
        for i in range(40):
            other = Record(None, str(i))
            ledger.refresh_metadata_sizes()
            self.assertEqual(ledger.total, footprint(records))

    def test_append_recent_promotion_commit_and_noop(self):
        cache = make_cache()
        self.check(cache)
        for i in range(12):
            append(cache, i)
            self.check(cache)
        self.assertEqual(set(cache._managed), {10, 11})
        cache.recent_frames = 4
        self.assertEqual(cache.recent_ids, (8, 9, 10, 11))
        self.check(cache)
        self.assertEqual(set(cache._managed), {8, 9, 10, 11})
        result = cache.close_segment([torch.ones(2, 4, 16)]*2)
        self.assertEqual(result['last_event']['after_bytes'], self.check(cache)['persistent_bytes'])
        self.assertEqual(len(cache._managed), 12)
        self.assertEqual(cache.close_segment(None)['persistent_bytes'], result['persistent_bytes'])

    def test_shared_views_packed_shrinking_and_snapshots(self):
        cache = make_cache(frames=6)
        cache.close_segment([torch.ones(2, 4, 16)]*2)
        first = cache.frames[0]
        # Same tensor object, distinct views of one storage, and shared layers.
        layer = first.layers[0]
        layer = replace(layer, high_v=layer.high_k.view_as(layer.high_k), low_v=layer.low_k)
        cache.frames[0] = replace(first, layers=(layer, layer))
        cache.frames[1] = replace(cache.frames[1], layers=cache.frames[0].layers)
        self.check(cache)
        snapshot = cache.acquire_snapshot(1)
        self.check(cache)
        before = cache.frames[0].layers[0].materialize()[0].clone()
        for _ in range(3):
            for i in (0, 1):
                frame = cache.frames[i]
                cache.frames[i] = replace(frame, layers=tuple(x.shrink(8) for x in frame.layers))
            self.check(cache)
        cache.record_activation(1, {0, 1})
        self.check(cache)
        for i in (0, 1):
            frame = cache.frames[i]
            cache.frames[i] = replace(frame, layers=tuple(x.drop_low() for x in frame.layers))
            self.check(cache)
            del cache.frames[i]
            self.check(cache)
        torch.testing.assert_close(snapshot.frames[0].layers[0].materialize()[0], before)
        pinned = self.check(cache)['cache_working_bytes']
        cache.release_snapshot(1)
        self.assertLess(self.check(cache)['cache_working_bytes'], pinned)

    def test_public_mapping_operations_and_scalar_identity_reuse(self):
        cache = make_cache(frames=6)
        cache.close_uncompressed_segment()
        saved = dict(cache.frames)
        cache.frames.pop(0)
        self.check(cache)
        cache.frames.update({0: saved[0]})
        cache.frames |= {1: replace(saved[1], hits=7)}
        self.check(cache)
        cache.frames.popitem()
        self.check(cache)
        cache.frames = saved
        self.check(cache)
        for i in range(250, 360):
            cache.counts['compressions'] = i
            cache.last_event['elapsed_seconds'] = float(i)
            cache.last_event['after_bytes'] = int(str(i*10))
            cache.version = i
            self.check(cache)
        cache.last_event = {'reason': 'replacement', 'number': 999}
        cache.counts = dict(cache.counts, compressions=1000)
        self.check(cache)
        cache.acquire_snapshot(1)
        cache.snapshots.clear()
        self.check(cache)
        cache.frames.clear()
        self.check(cache)

    def test_no_tensor_retention_after_eviction_or_snapshot_release(self):
        cache = make_cache(frames=3)
        cache.close_uncompressed_segment()
        tensor = weakref.ref(cache.frames[0].layers[0].high_k)
        cache.acquire_snapshot(1)
        del cache.frames[0]
        self.assertIsNotNone(tensor())
        cache.release_snapshot(1)
        gc.collect()
        self.assertIsNone(tensor())
        self.check(cache)
        tensor = weakref.ref(cache.frames[1].layers[0].high_k)
        del cache.frames[1]
        gc.collect()
        self.assertIsNone(tensor())

    def test_discarding_cache_releases_tensors_without_garbage_collection(self):
        cache = make_cache(frames=3)
        tensor = weakref.ref(cache.frames[0].layers[0].high_k)
        cache_ref = weakref.ref(cache)
        gc.disable()
        try:
            del cache
            self.assertIsNone(cache_ref())
            self.assertIsNone(tensor())
        finally:
            gc.enable()

    def test_stats_do_not_walk_frames_or_tensor_storages(self):
        cache = make_cache(frames=8)
        cache.close_uncompressed_segment()
        cache.memory_stats()
        with patch('model.suprkv.cache.footprint', side_effect=AssertionError('Full scan')), \
                patch('model.suprkv.cache.tensor_storage_bytes', side_effect=AssertionError('Full scan')), \
                patch.object(torch.Tensor, 'untyped_storage', side_effect=AssertionError('Storage scan')):
            for _ in range(10):
                cache.persistent_bytes()
                cache.memory_stats()
                cache._enforce_budget()
            cache.record_activation(cache.acquire_snapshot(1).question_id, {0})
            cache.release_snapshot(1)
            cache.memory_stats()
        self.check(cache)

    def test_budget_policy_matches_full_scan_reference(self):
        class FullScanCache(StreamingKVCache):
            def persistent_bytes(self):
                return footprint(self._persistent())

        fast = make_cache()
        slow = FullScanCache(fast.sink, 100000, 8, group_size=8)
        for i in range(12):
            append(fast, i)
            append(slow, i)
        queries = [torch.ones(2, 4, 16)]*2
        fast.close_segment(queries)
        slow.close_segment(queries)
        for _ in range(6):
            # Force low-region deletion, progressive high shrinking, then eviction.
            budget = min(fast.persistent_bytes(), slow.persistent_bytes())-500
            fast.budget = slow.budget = budget
            fast._enforce_budget()
            slow._enforce_budget()
            self.check(fast)
            self.assertEqual(list(fast.frames), list(slow.frames))
            self.assertEqual(dict(fast.counts), dict(slow.counts))
            self.assertLessEqual(fast.persistent_bytes(), budget)
            for a, b in zip(fast.frames.values(), slow.frames.values()):
                for la, lb in zip(a.layers, b.layers):
                    self.assertEqual(la.bits, lb.bits)
                    torch.testing.assert_close(la.materialize()[0], lb.materialize()[0])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_payload_and_snapshot_ledger(self):
        cache = make_cache('cuda', frames=5)
        cache.close_segment([torch.ones(2, 4, 16, device='cuda')]*2)
        snapshot = cache.acquire_snapshot(1)
        for i in range(3):
            frame = cache.frames[i]
            cache.frames[i] = replace(frame, layers=tuple(x.shrink(8) for x in frame.layers))
        stats = self.check(cache)
        self.assertEqual(stats['persistent_cuda_bytes'], stats['persistent_tensor_bytes'])
        cache.release_snapshot(1)
        self.check(cache)


if __name__ == '__main__':
    unittest.main()
