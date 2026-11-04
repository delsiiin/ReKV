"""P0 CUDA kernels against independent dense scores and stable partitions.

Without a GPU, exercise the actual kernels (not their torch fallbacks) with:
TRITON_INTERPRET=1 python -m unittest discover -s tests -p 'test_suprkv_scoring_fusion.py' -v
The interpreter checks arithmetic/addressing; it does not validate GPU execution.
"""
from contextlib import ExitStack
from dataclasses import replace
import importlib.util
import os
import unittest
from unittest.mock import patch

import torch

from model.suprkv.cache import LayerFrame, StreamingKVCache, _fused_partition_supported
from model.suprkv.segments import segment_frame_query_scores, _fused_frame_scores_supported


INTERPRETER = os.environ.get('TRITON_INTERPRET') == '1'
FIELDS = ('high_indices', 'low_indices', 'high_k', 'high_v', 'low_k', 'low_v')


def dense_reference(query, keys):
    """Explicit GQA expansion and FP64 QK; no pooled-query shortcut."""
    frames, (heads, tokens, dim) = len(keys), keys[0].shape
    group = query.shape[1] // heads
    q = query.double().permute(1, 0, 2)
    k = torch.cat(keys, 1).double().repeat_interleave(group, 0)
    logits = (q @ k.transpose(-1, -2)) / dim**.5
    return logits.mean(1).reshape(heads, group, frames, tokens).mean(1).permute(1, 0, 2)


class FusionFallbackTests(unittest.TestCase):
    def test_cpu_batch_preserves_reference_and_rejects_malformed_shapes(self):
        entries = [LayerFrame.full(torch.randn(2, 7, 8), torch.randn(2, 7, 8)) for _ in range(3)]
        scores = torch.randn(3, 2, 7)
        self.assertFalse(_fused_partition_supported(entries, scores))
        self.assertFalse(_fused_frame_scores_supported(torch.randn(3, 4, 8), [e.high_k for e in entries]))
        for actual, entry, row in zip(LayerFrame.classify_batch(entries, scores), entries, scores):
            reference = entry.classify_scores(row)
            for name in FIELDS:
                torch.testing.assert_close(getattr(actual, name), getattr(reference, name), rtol=0, atol=0)
        for bad in (scores[0], scores[:1], scores[:, :, :2]):
            with self.assertRaises(ValueError):
                LayerFrame.classify_batch(entries, bad)


@unittest.skipUnless((torch.cuda.is_available() or INTERPRETER) and importlib.util.find_spec('triton'),
                     'CUDA or TRITON_INTERPRET=1 required')
class ScoringFusionTests(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu' if INTERPRETER else 'cuda'
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.context.enter_context(torch.inference_mode())
        if INTERPRETER:
            # Bypass only device eligibility; execute the real dispatch/wrappers/kernels.
            self.context.enter_context(patch('model.suprkv.segments._fused_frame_scores_supported', return_value=True))
            self.context.enter_context(patch('model.suprkv.cache._fused_partition_supported', return_value=True))

    def assert_partition_equal(self, actual, expected):
        for name in FIELDS:
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
        self.assertEqual(actual.high_offsets, expected.high_offsets)
        self.assertEqual(actual.low_offsets, expected.low_offsets)

    def test_tiled_scores_gqa_tail_tiles_and_noncontiguous_inputs(self):
        torch.manual_seed(608)
        dtypes = (torch.float16, torch.bfloat16, torch.float32)
        for dtype in dtypes:
            for frames, heads, group, tokens, dim in (
                    (1, 1, 1, 1, 1), (3, 2, 3, 7, 8), (33, 1, 2, 3, 40),
                    (4, 4, 7, 33, 128), (2, 2, 1, 196, 256)):
                with self.subTest(dtype=dtype, shape=(frames, heads, group, tokens, dim)):
                    q = torch.randn(frames, heads*group, dim*2, dtype=dtype, device=self.device)[..., ::2]
                    keys = list(torch.randn(frames, heads, tokens, dim*2, dtype=dtype, device=self.device)[..., ::2])
                    reference = dense_reference(q, keys)
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                        actual = segment_frame_query_scores(q, keys, headwise=True)
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual.double(), reference, rtol=1e-5, atol=1e-6)
                    # Also compare the previous FP32 operation order, including masks.
                    with patch('model.suprkv.segments._fused_frame_scores_supported', return_value=False):
                        previous = segment_frame_query_scores(q, keys, headwise=True)
                    torch.testing.assert_close(actual, previous, rtol=1e-5, atol=1e-6)
                    self.assertTrue(torch.equal(actual >= actual.mean(-1, keepdim=True),
                                                previous >= previous.mean(-1, keepdim=True)))

    def test_partition_exact_payload_order_ties_and_independent_storage(self):
        torch.manual_seed(17)
        dtypes = (torch.float16, torch.bfloat16, torch.float32)
        for dtype in dtypes:
            for tokens in (1, 7, 33, 196):
                dim = 11  # Scatter blocks cross both token and head boundaries.
                entries = [LayerFrame.full(torch.randn(3, tokens, dim, device=self.device),
                                           torch.randn(3, tokens, dim, device=self.device), dtype) for _ in range(3)]
                entries = [replace(e, high_indices=e.high_indices*3+7) for e in entries]
                scores = torch.randn(3, 3, tokens, device=self.device)
                scores[0] = 0  # All-high frame, including a zero-sized low buffer.
                scores[1, 0] = 1  # Empty low for only one head.
                scores[2, 1] = 0
                if tokens > 1:
                    scores[2, 1, 0], scores[2, 1, -1] = -2, 2  # Exact mean ties.
                expected = [e.classify_scores(s) for e, s in zip(entries, scores)]
                with patch.object(LayerFrame, 'classify_scores', side_effect=AssertionError('per-frame partition')):
                    actual = LayerFrame.classify_batch(entries, scores)
                storages = set()
                for e, a, ref in zip(entries, actual, expected):
                    self.assert_partition_equal(a, ref)
                    for name in FIELDS:
                        x = getattr(a, name)
                        self.assertEqual(x.untyped_storage().nbytes(), x.numel()*x.element_size())
                        if x.numel():
                            self.assertNotIn(x.data_ptr(), storages)
                            storages.add(x.data_ptr())
                    original = e.high_k.clone()
                    a.high_k.zero_()
                    torch.testing.assert_close(e.high_k, original, rtol=0, atol=0)

    def test_nonfinite_inputs_and_overflow_are_rejected(self):
        q = torch.ones(3, 4, 8, device=self.device)
        keys = list(torch.ones(3, 2, 7, 8, device=self.device))
        for bad in (float('nan'), float('inf'), -float('inf')):
            for target in ('query', 'key'):
                query, kk = q.clone(), [k.clone() for k in keys]
                if target == 'query':
                    query[-1, -1, -1] = bad
                else:
                    kk[-1][-1, -1, -1] = bad
                with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                    segment_frame_query_scores(query, kk, headwise=True)
            entries = [LayerFrame.full(k, k, torch.float32) for k in keys]
            scores = torch.ones(3, 2, 7, device=self.device)
            scores[-1, -1, -1] = bad
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                LayerFrame.classify_batch(entries, scores)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            segment_frame_query_scores(q*1e30, [k*1e30 for k in keys], headwise=True)

    def test_signed_logits_and_exact_near_mean_ties(self):
        # Opposing queries distinguish raw-logit averaging from softmax or abs.
        # Binary fractions keep threshold ties exactly representable.
        values = torch.tensor([-1., -2**-20, 0., 2**-20, 1.], device=self.device)
        key = values.reshape(1, 5, 1).expand(2, 5, 4).contiguous()
        q = torch.stack((torch.full((4, 4), 2., device=self.device),
                         torch.full((4, 4), -1., device=self.device)))
        keys = [key, -key]
        scores = segment_frame_query_scores(q, keys, headwise=True)
        reference = dense_reference(q, keys).float()
        torch.testing.assert_close(scores, reference, rtol=0, atol=0)
        entries = [LayerFrame.full(k, k+1, torch.float32) for k in keys]
        for actual, entry, row in zip(LayerFrame.classify_batch(entries, scores), entries, reference):
            self.assert_partition_equal(actual, entry.classify_scores(row))
        zeros = segment_frame_query_scores(torch.zeros_like(q), keys, headwise=True)
        for entry in LayerFrame.classify_batch(entries, zeros):
            self.assertEqual(entry.high_offsets, (0, 5, 10))
            self.assertEqual(entry.low_indices.numel(), 0)

    def make_cache(self, failure=False):
        torch.manual_seed(62)
        sink = torch.zeros(2, 2, 16, device=self.device)
        cache = StreamingKVCache([(sink, sink)]*2, 100000, 7, dtype=torch.float32,
                                 recent_frames=1, group_size=8, compression_granularity='head')
        for frame in range(3):
            keys = list(torch.randn(2, 2, 7, 16, device=self.device))
            queries = list(torch.randn(2, 6, 16, device=self.device))
            if failure and frame == 2:
                queries[-1][-1, -1] = float('nan')
            cache.append_frame(frame, [(k+10, k-10) for k in keys], score_keys=keys, frame_queries=queries)
        return cache

    def test_commit_snapshot_budget_and_quantization_match_reference(self):
        actual, reference = self.make_cache(), self.make_cache()
        snapshot = actual.acquire_snapshot(1)
        original_layers = [f.layers for f in snapshot.frames]
        actual.close_segment(None, scoring_strategy='frame_mean_query')
        with patch('model.suprkv.segments._fused_frame_scores_supported', return_value=False), \
             patch('model.suprkv.cache._fused_partition_supported', return_value=False):
            reference.close_segment(None, scoring_strategy='frame_mean_query')
        self.assertEqual(actual.counts['scoring_failures'], 0)
        for f, r in zip(actual.frames.values(), reference.frames.values()):
            for a, b in zip(f.layers, r.layers):
                self.assert_partition_equal(a, b)
        for f, original in zip(snapshot.frames, original_layers):
            self.assertIs(f.layers, original)
            self.assertFalse(f.closed)
        actual.audit_accounting()
        actual.release_snapshot(1)
        self.assertEqual(actual.persistent_bytes(), reference.persistent_bytes())
        for cache in (actual, reference):
            # Force pruning then a precision step; retain the recent frame.
            cache.budget = cache.persistent_bytes() // 2
            cache._enforce_budget()
            cache.audit_accounting()
            self.assertFalse(cache._score_keys)
            self.assertFalse(cache._score_queries)
        self.assertEqual(dict(actual.counts), dict(reference.counts))
        self.assertEqual(actual.persistent_bytes(), reference.persistent_bytes())
        self.assertEqual(tuple(actual.frames), tuple(reference.frames))
        for f, r in zip(actual.frames.values(), reference.frames.values()):
            for a, b in zip(f.layers, r.layers):
                self.assertEqual(a.bits, b.bits)
                for head in range(a.heads):
                    for x, y in zip(a.materialize_head(head), b.materialize_head(head)):
                        torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_late_layer_failure_is_atomic_and_releases_working_memory(self):
        cache = self.make_cache(failure=True)
        originals = {i: f.layers for i, f in cache.frames.items()}
        cache.close_segment(None, scoring_strategy='frame_mean_query')
        self.assertEqual(cache.counts['scoring_failures'], 1)
        for i, frame in cache.frames.items():
            self.assertIs(frame.layers, originals[i])
            self.assertTrue(frame.closed)
            self.assertIn('Nonfinite', frame.score_error)
        self.assertFalse(cache._score_keys)
        self.assertFalse(cache._score_queries)
        cache.audit_accounting()


if __name__ == '__main__':
    unittest.main()
