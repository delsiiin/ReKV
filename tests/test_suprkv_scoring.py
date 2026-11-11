"""Segment batching must preserve frame partitions and failure atomicity."""
import unittest

import torch

from model.suprkv.cache import LayerFrame, StreamingKVCache
from model.suprkv.segments import (segment_summary_scores, summary_scores, segment_frame_query_scores,
                                   frame_local_query_scores)


class SegmentScoringTests(unittest.TestCase):
    def test_layer_frame_scores_match_dense_gqa_reference_and_independent_batches(self):
        torch.manual_seed(118)
        for device in self.devices():
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                with self.subTest(device=device, dtype=dtype):
                    queries = torch.randn(4, 6, 8, device=device, dtype=dtype)
                    keys = list(torch.randn(4, 2, 7, 8, device=device, dtype=dtype))
                    # Expand GQA explicitly and average all query heads after QK.
                    logits = (queries.double().unsqueeze(-2) @
                              torch.stack(keys).double().repeat_interleave(3, dim=1).transpose(-1, -2)) / 8**.5
                    expected = logits.squeeze(-2).mean(1)
                    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
                        scores = frame_local_query_scores(queries, keys, headwise=False)
                    self.assertEqual(scores.dtype, torch.float32)
                    torch.testing.assert_close(scores.double(), expected, rtol=1e-5, atol=1e-6)
                    singles = torch.cat([frame_local_query_scores(queries[i:i+1], [keys[i]], headwise=False)
                                         for i in range(4)])
                    torch.testing.assert_close(scores, singles, rtol=1e-5, atol=1e-6)
                    changed = queries.clone()
                    changed[1:] *= -100
                    keys[1:] = [k * 100 for k in keys[1:]]
                    torch.testing.assert_close(frame_local_query_scores(changed, keys, headwise=False)[0],
                                               scores[0], rtol=0, atol=0)

    def test_layer_frame_commit_uses_shared_masks_and_preserves_frame_queries(self):
        # Heads disagree and opposite frame queries cancel only in segment mode.
        key = torch.tensor([[[-2.], [0.], [2.]], [[4.], [0.], [-4.]]])
        for scope in ('segment', 'frame'):
            for batch_size in (1, 2) if scope == 'frame' else (2,):
                with self.subTest(scope=scope, batch_size=batch_size):
                    cache = StreamingKVCache([(key[:, :1], key[:, :1])], 100000, 3,
                                             dtype=torch.float32, compression_granularity='layer', scoring_scope=scope)
                    for i, sign in enumerate((1., -1.)):
                        cache.append_frame(i, [(key, key)], score_keys=[key],
                                           frame_queries=[torch.full((4, 1), sign)])
                        if (i + 1) % batch_size == 0:
                            cache.close_segment(None, scoring_strategy='frame_mean_query')
                    for i in range(2):
                        entry = cache.frames[i].layers[0]
                        expected = [0, 1, 2] if scope == 'segment' else ([0, 1] if i == 0 else [1, 2])
                        self.assertIsNone(entry.high_offsets)
                        self.assertEqual(entry.high_indices.tolist(), expected)
                    self.assertEqual(cache.counts['scoring_failures'], 0)
                    self.assertFalse(cache._score_queries or cache._score_keys)
                    cache.audit_accounting()

    def test_frame_queries_match_dense_logits_and_working_memory_lifecycle(self):
        torch.manual_seed(77)
        sink = torch.zeros(2, 2, 4)
        cache = StreamingKVCache([(sink, sink)]*2, 100000, 3, dtype=torch.float32, compression_granularity="head", scoring_scope="frame")
        queries = torch.randn(2, 3, 4, 4)  # layer, frame, head, dim
        keys = torch.randn(2, 3, 2, 3, 4)
        for frame in range(3):
            cache.append_frame(frame, [(k, k) for k in keys[:, frame]],
                               score_keys=list(keys[:, frame]), frame_queries=list(queries[:, frame]))
        before = cache.memory_stats()
        cache.audit_accounting()
        qrefs = list(cache._score_queries.values())
        cache.clear_frame_queries()
        without = cache.memory_stats()
        self.assertEqual(before['persistent_tensor_bytes'], without['persistent_tensor_bytes'])
        self.assertGreater(before['cache_working_bytes'], without['cache_working_bytes'])
        for frame, row in enumerate(qrefs):
            cache._score_queries[frame] = row
        cache.close_segment(None, scoring_strategy='frame_mean_query')
        for layer in range(2):
            q = queries[layer].double().unsqueeze(-2)
            k = keys[layer].double().repeat_interleave(2, dim=1)
            scores = (q @ k.transpose(-1, -2) / 2).reshape(3, 2, 2, 3).mean(2)
            for frame in range(3):
                entry = cache.frames[frame].layers[layer]
                for head in range(2):
                    row = scores[frame, head]
                    expected = torch.where(row >= row.mean())[0].tolist()
                    start, end = entry.high_offsets[head:head+2]
                    self.assertEqual(entry.high_indices[start:end].tolist(), expected)
        self.assertFalse(cache._score_queries)
        self.assertFalse(cache._score_keys)
        cache.audit_accounting()

    def test_frame_local_scores_are_independent_of_batch_members(self):
        torch.manual_seed(117)
        for device in self.devices():
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                queries = torch.randn(4, 6, 8, device=device, dtype=dtype)
                keys = list(torch.randn(4, 2, 7, 8, device=device, dtype=dtype))
                with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
                    scores = frame_local_query_scores(queries, keys)
                q = queries.double().unsqueeze(-2)
                k = torch.stack(keys).double().repeat_interleave(3, dim=1)
                expected = (q @ k.transpose(-1, -2) / 8**.5).reshape(4, 2, 3, 7).mean(2)
                self.assertEqual(scores.dtype, torch.float32)
                torch.testing.assert_close(scores.double(), expected, rtol=1e-5, atol=1e-6)
                singles = torch.cat([frame_local_query_scores(queries[i:i+1], [keys[i]]) for i in range(4)])
                torch.testing.assert_close(scores, singles, rtol=1e-5, atol=1e-6)
                changed = queries.clone()
                changed[1:] *= -100
                keys[1:] = [k * 100 for k in keys[1:]]
                torch.testing.assert_close(frame_local_query_scores(changed, keys)[0], scores[0], rtol=0, atol=0)

    def test_opposite_frame_queries_do_not_cancel_during_cache_commit(self):
        # Segment pooling would cancel the queries and mark every token high.
        key = torch.tensor([[[-2.], [0.], [2.]]])
        cache = StreamingKVCache([(key[:, :1], key[:, :1])], 100000, 3,
                                 dtype=torch.float32, compression_granularity='head', scoring_scope='frame')
        for i, sign in enumerate((1., -1.)):
            cache.append_frame(i, [(key, key)], score_keys=[key], frame_queries=[torch.tensor([[sign]])])
        cache.close_segment(None, scoring_strategy='frame_mean_query')
        self.assertEqual(cache.frames[0].layers[0].high_indices.tolist(), [1, 2])
        self.assertEqual(cache.frames[1].layers[0].high_indices.tolist(), [0, 1])
        self.assertEqual(cache.last_event['trigger'], 'frame_batch_commit')
        # The default must retain the original cross-frame query averaging.
        segment = StreamingKVCache([(key[:, :1], key[:, :1])], 100000, 3,
                                   dtype=torch.float32, compression_granularity='head')
        for i, sign in enumerate((1., -1.)):
            segment.append_frame(i, [(key, key)], score_keys=[key], frame_queries=[torch.tensor([[sign]])])
        segment.close_segment(None, scoring_strategy='frame_mean_query')
        for frame in segment.frames.values():
            self.assertEqual(frame.layers[0].high_indices.tolist(), [0, 1, 2])
        self.assertEqual(segment.last_event['trigger'], 'segment_commit')
        self.assertEqual(segment.memory_stats()['scoring_scope'], 'segment')

    def test_frame_query_missing_or_nonfinite_preserves_entire_segment(self):
        key = torch.ones(2, 3, 4)
        for invalid in ('missing', 'nonfinite'):
            cache = StreamingKVCache([(key, key)], 100000, 3, compression_granularity="head")
            q = torch.full((4, 4), float('nan')) if invalid == 'nonfinite' else None
            cache.append_frame(0, [(key, key)], score_keys=[key], frame_queries=[q] if q is not None else None)
            cache.close_segment(None, scoring_strategy='frame_mean_query')
            self.assertEqual(cache.counts['scoring_failures'], 1)
            self.assertEqual(cache.frames[0].layers[0].high_indices.tolist(), [0, 1, 2])
            self.assertFalse(cache._score_queries)
            cache.audit_accounting()

    def devices(self):
        return ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])

    def test_frame_attention_matches_fp64_gqa_reference(self):
        torch.manual_seed(123)
        for device in self.devices():
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                for frames in (1, 2, 8, 32):
                    with self.subTest(device=device, dtype=dtype, frames=frames):
                        query = torch.randn(frames, 6, 8, device=device, dtype=dtype)
                        keys = list(torch.randn(frames, 2, 7, 8, device=device, dtype=dtype))
                        q = query.double().permute(1, 0, 2)
                        k = torch.cat(keys, dim=1).double().repeat_interleave(3, dim=0)
                        reference = (q @ k.transpose(-1, -2) / 8**.5).mean(1).reshape(2, 3, frames, 7).mean(1).permute(1, 0, 2)
                        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
                            actual = segment_frame_query_scores(query, keys, headwise=True)
                        self.assertEqual(actual.dtype, torch.float32)
                        torch.testing.assert_close(actual.double(), reference, rtol=1e-5, atol=1e-6)
                        torch.testing.assert_close(actual, segment_summary_scores(query, keys, headwise=True), rtol=1e-5, atol=1e-6)

    def test_frame_scores_use_logits_instead_of_probabilities(self):
        key = torch.tensor([-1., 0., 1.]).reshape(1, 3, 1)
        # Averaged logits keep the zero and positive endpoints, whereas
        # averaged softmax probabilities would keep both nonzero endpoints.
        for query in (torch.tensor([2., -1.]).reshape(2, 1, 1),
                      torch.tensor([2., -1.]).reshape(1, 2, 1)):
            keys = [key] * len(query)
            scores = segment_frame_query_scores(query, keys, headwise=True)
            for row in scores[:, 0]:
                torch.testing.assert_close(row, torch.tensor([-.5, 0., .5]))
                self.assertEqual(torch.where(row >= row.mean())[0].tolist(), [1, 2])

    def test_frame_scores_are_unnormalized_and_handle_ties(self):
        query = torch.ones(2, 1, 1)
        keys = [torch.zeros(1, 3, 1), torch.ones(1, 3, 1)]
        scores = segment_frame_query_scores(query, keys, headwise=True)
        torch.testing.assert_close(scores, torch.tensor([[[0., 0., 0.]], [[1., 1., 1.]]]))
        uniform = segment_frame_query_scores(torch.zeros_like(query), keys, headwise=True)
        torch.testing.assert_close(uniform, torch.zeros(2, 1, 3))
        self.assertTrue((uniform >= uniform.mean(-1, keepdim=True)).all())

    def test_frame_attention_rejects_invalid_inputs(self):
        query = torch.ones(2, 4, 8)
        key = torch.ones(2, 3, 8)
        for invalid in (float('nan'), float('inf')):
            bad_key, bad_query = key.clone(), query.clone()
            bad_key[0, 0, 0] = invalid
            bad_query[0, 0, 0] = invalid
            for q, keys in ((query, [key, bad_key]), (bad_query, [key, key])):
                with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                    segment_frame_query_scores(q, keys, headwise=True)
        for q, keys in ((query[:0], []), (query[:1], [key, key]),
                        (query[:, :3], [key, key]), (query, [key, key[:, :2]])):
            with self.assertRaises(ValueError):
                segment_frame_query_scores(q, keys, headwise=True)

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
                        expected = torch.stack([summary_scores(query, key, headwise=True) for key in keys])
                        actual = segment_summary_scores(query, keys, headwise=True)
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
                torch.einsum('shd,hnd->shn', query.double(), key.double().repeat_interleave(3, 0))
                .mean(0).reshape(2, 3, 7).mean(1) / 8**.5 for key in keys])
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
                actual = segment_summary_scores(query, keys, headwise=True)
                single = summary_scores(query, keys[0], headwise=True)
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
            scores = segment_summary_scores(query, [key, key + 10], headwise=True)
            for source, row in zip((key, key + 10), scores):
                layer = LayerFrame.full(source, source + 1)
                split = layer.classify_scores(row)
                self.assertEqual(split.high_indices.tolist(), [1, 2, 3, 1, 2, 3])
                self.assertEqual(split.high_offsets, (0, 3, 6))
                k, v, indices = split.materialize()
                torch.testing.assert_close(k, layer.high_k, rtol=0, atol=0)
                torch.testing.assert_close(v, layer.high_v, rtol=0, atol=0)
                self.assertEqual(indices.tolist(), [0, 1, 2, 3])
                split.high_k.zero_()
                torch.testing.assert_close(layer.high_k, source.half(), rtol=0, atol=0)
            zeros = segment_summary_scores(torch.zeros_like(query), [key, key], headwise=True)
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
                        segment_summary_scores(q, keys, headwise=True)
            for q, keys in ((query[:0], [key]), (query[:, :3], [key]),
                            (query, []), (query, [key, key[:, :2]])):
                with self.assertRaises(ValueError):
                    segment_summary_scores(q, keys, headwise=True)

    def test_commit_matches_individual_scoring_and_preserves_snapshot(self):
        torch.manual_seed(41)
        for device in self.devices():
            sink = torch.zeros(2, 2, 8, device=device)
            cache = StreamingKVCache([(sink, sink)] * 3, 100000, 7, compression_granularity="head")
            queries = [torch.randn(5, 4, 8, device=device) for _ in range(3)]
            expected = {}
            for i in range(4):
                keys = [torch.randn(2, 7, 8, device=device) for _ in range(3)]
                cache.append_frame(i, [(key + 10, key - 10) for key in keys], score_keys=keys)
                expected[i] = [layer.classify(q, key, headwise=True) for layer, q, key in
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
                    cache = StreamingKVCache([(key[:, :2], key[:, :2])] * 2, 100000, 7, compression_granularity="head")
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
