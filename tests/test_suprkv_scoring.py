"""Segment batching must preserve frame partitions and failure atomicity."""
import unittest

import torch

from model.suprkv.cache import LayerFrame, StreamingKVCache
from model.suprkv.segments import segment_summary_scores, summary_scores


class SegmentScoringTests(unittest.TestCase):
    def devices(self):
        return ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])

    def test_batched_scores_and_masks_match_individual_frames(self):
        torch.manual_seed(72)
        for device in self.devices():
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                for frames in (1, 2, 8, 32):
                    with self.subTest(device=device, dtype=dtype, frames=frames):
                        query = torch.randn(19, 28, 128, device=device, dtype=dtype)
                        keys = list(torch.randn(frames, 4, 196, 128, device=device, dtype=dtype))
                        # Distinct offsets catch accidental use of a segment-wide mean.
                        keys = [key + i for i, key in enumerate(keys)]
                        expected = torch.stack([summary_scores(query, key) for key in keys])
                        actual = segment_summary_scores(query, keys)
                        self.assertEqual(actual.dtype, torch.float32)
                        self.assertEqual(expected.dtype, torch.float32)
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                        self.assertTrue(torch.equal(actual >= actual.mean(-1, keepdim=True),
                                                    expected >= expected.mean(-1, keepdim=True)))

    def test_fp32_scores_match_fp64_reference_under_autocast(self):
        torch.manual_seed(13)
        for device in self.devices():
            query = torch.randn(5, 6, 8, device=device)
            keys = list(torch.randn(4, 2, 7, 8, device=device))
            # Independent reference: explicitly expand GQA and keep text tokens.
            reference = torch.stack([
                torch.einsum('shd,hnd->sn', query.double(), key.double().repeat_interleave(3, 0)).mean(0)
                / (6 * 8**.5) for key in keys])
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
                actual = segment_summary_scores(query, keys)
                single = summary_scores(query, keys[0])
            self.assertEqual(actual.dtype, torch.float32)
            self.assertEqual(single.dtype, torch.float32)
            torch.testing.assert_close(actual.double(), reference, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(single.double(), reference[0], rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.equal(actual >= actual.mean(-1, keepdim=True),
                                        reference >= reference.mean(-1, keepdim=True)))

    def test_ties_all_high_and_partition_storage_is_independent(self):
        for device in self.devices():
            # sqrt(head_dim)=2 keeps these exact ties representable after scaling.
            key = torch.tensor([-2., 0., 0., 2.], device=device).reshape(1, 4, 1).expand(2, 4, 4)
            query = torch.ones(3, 4, 4, device=device)
            scores = segment_summary_scores(query, [key, key + 10])
            for source, row in zip((key, key + 10), scores):
                layer = LayerFrame.full(source, source + 1)
                split = layer.classify_scores(row)
                self.assertEqual(split.high_indices.tolist(), [1, 2, 3])
                k, v, indices = split.materialize()
                torch.testing.assert_close(k, layer.high_k, rtol=0, atol=0)
                torch.testing.assert_close(v, layer.high_v, rtol=0, atol=0)
                self.assertEqual(indices.tolist(), [0, 1, 2, 3])
                split.high_k.zero_()
                torch.testing.assert_close(layer.high_k, source.half(), rtol=0, atol=0)
            zeros = segment_summary_scores(torch.zeros_like(query), [key, key])
            self.assertTrue((zeros >= zeros.mean(-1, keepdim=True)).all())

    def test_invalid_batch_is_rejected(self):
        for device in self.devices():
            query = torch.ones(3, 4, 8, device=device)
            key = torch.ones(2, 4, 8, device=device)
            for invalid in (float('nan'), float('inf')):
                bad_key, bad_query = key.clone(), query.clone()
                bad_key[0, 0, 0] = invalid
                bad_query[0, 0, 0] = invalid
                for q, keys in ((query, [key, bad_key]), (bad_query, [key, key])):
                    with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                        segment_summary_scores(q, keys)
            for q, keys in ((query[:0], [key]), (query[:, :3], [key]),
                            (query, []), (query, [key, key[:, :2]])):
                with self.assertRaises(ValueError):
                    segment_summary_scores(q, keys)

    def test_commit_matches_individual_scoring_and_preserves_snapshot(self):
        torch.manual_seed(41)
        for device in self.devices():
            sink = torch.zeros(2, 2, 8, device=device)
            cache = StreamingKVCache([(sink, sink)] * 3, 100000, 7)
            queries = [torch.randn(5, 4, 8, device=device) for _ in range(3)]
            expected = {}
            for i in range(4):
                keys = [torch.randn(2, 7, 8, device=device) for _ in range(3)]
                cache.append_frame(i, [(key + 10, key - 10) for key in keys], score_keys=keys)
                expected[i] = [layer.classify(q, key) for layer, q, key in
                               zip(cache.frames[i].layers, queries, cache._score_keys[i])]
            snapshot = cache.acquire_snapshot(1)
            originals = tuple(snapshot.frames)
            cache.close_segment(queries)
            for i, frame in cache.frames.items():
                for actual, reference in zip(frame.layers, expected[i]):
                    for field in ('high_indices', 'low_indices', 'high_k', 'high_v', 'low_k', 'low_v'):
                        torch.testing.assert_close(getattr(actual, field), getattr(reference, field), rtol=0, atol=0)
            self.assertEqual(tuple(snapshot.frames), originals)
            self.assertTrue(all(not frame.closed for frame in snapshot.frames))
            self.assertFalse(cache._score_keys)
            cache.audit_accounting()
            cache.release_snapshot(1)
            cache.audit_accounting()

    def test_late_layer_failure_falls_back_for_entire_segment(self):
        for device in self.devices():
            for failure in ('query', 'key', 'missing'):
                with self.subTest(device=device, failure=failure):
                    key = torch.randn(2, 7, 8, device=device)
                    cache = StreamingKVCache([(key[:, :2], key[:, :2])] * 2, 100000, 7)
                    for i in range(3):
                        cache.append_frame(i, [(key, key)] * 2,
                                           score_keys=None if failure == 'missing' and i == 2 else [key, key])
                    originals = dict(cache.frames)
                    queries = [torch.ones(3, 4, 8, device=device) for _ in range(2)]
                    if failure == 'query':
                        queries[-1][0, 0, 0] = float('nan')
                    elif failure == 'key':
                        cache._score_keys[2][-1][0, 0, 0] = float('inf')
                    cache.close_segment(queries)
                    self.assertEqual(cache.counts['scoring_failures'], 1)
                    for i, frame in cache.frames.items():
                        self.assertIs(frame.layers, originals[i].layers)
                        self.assertTrue(frame.closed)
                        self.assertTrue(frame.score_error)
                    self.assertFalse(cache._score_keys)
                    cache.audit_accounting()


if __name__ == '__main__':
    unittest.main()
