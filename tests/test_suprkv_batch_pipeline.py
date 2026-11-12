"""Event-batched quantization, dirty-frame invalidation and prefix restoration."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from model.suprkv.cache import LayerFrame, StreamingKVCache
from model.suprkv.encoding import EncodingPrefix
from model.suprkv.batch_quantization import lower
from model.suprkv.prefix_batch import materialize_many
from model.suprkv.encoding_attention import EncodingSegments, attend


class InvalidationTests(unittest.TestCase):
    def test_only_changed_frames_are_checked_and_rebinding_is_safe(self):
        torch.set_num_threads(1)
        sink = torch.zeros(2, 2, 8)
        cache = StreamingKVCache([(sink, sink)], 10000, 16, dtype=torch.float32)
        for i in range(3):
            k = torch.randn(2, 16, 8)
            cache.append_frame(i, [(k, k)], score_keys=[k])
        prefix = EncodingPrefix()
        kwargs = dict(capture_scores=False, capture_frame_query=False)
        prefix.contexts(cache, **kwargs)
        with patch.object(cache.frames, 'get', wraps=cache.frames.get) as get:
            prefix.prune(cache.frames)
            self.assertEqual(get.call_count, 0)
        cache.frames[0] = replace(cache.frames[0], hits=1)
        with patch.object(cache.frames, 'get', wraps=cache.frames.get) as get:
            prefix.prune(cache.frames)
            self.assertEqual(get.call_count, 0)
        old = cache.frames[1]
        classified = old.layers[0].classify_scores(torch.randn(16))
        cache.frames[1] = replace(old, layers=(classified,))
        # Multiple changes to the same frame are coalesced into one check.
        cache.frames[1] = replace(cache.frames[1], layers=(classified.drop_low(),))
        with patch.object(cache.frames, 'get', wraps=cache.frames.get) as get:
            prefix.prune(cache.frames)
            self.assertEqual(get.call_count, 1)
        self.assertNotIn((1, 0), prefix.entries)
        self.assertIn((0, 0), prefix.entries)
        prefix.contexts(cache, **kwargs)
        del cache.frames[0]
        prefix.prune(cache.frames)
        self.assertNotIn((0, 0), prefix.entries)
        previous = cache.frames
        cache.frames = dict(cache.frames)
        prefix.contexts(cache, **kwargs)
        self.assertFalse(previous._listeners)
        prefix.prune(dict(cache.frames))
        self.assertFalse(cache.frames._listeners)
        prefix.contexts(cache, **kwargs)
        prefix.clear()
        self.assertFalse(cache.frames._listeners)
        self.assertEqual(prefix.working_bytes, 0)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class BatchCudaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(733)
        torch.set_num_threads(1)

    def entries(self, dtype, group):
        entries = []
        for compact in (False, True):
            for n in (17, 137, 196):
                k = torch.randn(3, n, 24, device='cuda', dtype=dtype)
                entry = LayerFrame.full(k, torch.randn_like(k), dtype)
                scores = torch.randn((3, n) if compact else (n,), device='cuda')
                entry = entry.classify_scores(scores)
                entries.append(entry)
                entry = entry.drop_low()
                entries.append(entry)
                for _ in range(3):
                    entry = entry.shrink(group)
                    entries.append(entry)
        return entries

    @torch.inference_mode()
    def test_quantized_payloads_and_restoration_are_exact(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for group in (8, 128):
                with self.subTest(dtype=dtype, group=group):
                    entries = self.entries(dtype, group)
                    expected = [e.shrink(group) for e in entries]
                    actual = lower(entries, group)
                    for old, a, b in zip(entries, actual, expected):
                        self.assertEqual(a.bits, b.bits)
                        self.assertIs(a.representative, old.representative)
                        if old.bits == 2:
                            self.assertIs(a, old)
                        for name in ('high_k', 'high_v'):
                            x, y = getattr(a, name), getattr(b, name)
                            for field in ('data', 'scale', 'offset'):
                                torch.testing.assert_close(getattr(x, field), getattr(y, field), rtol=0, atol=0)
                            for field in ('bits', 'group_size', 'axis'):
                                self.assertEqual(getattr(x, field), getattr(y, field))
                    restored = materialize_many(entries)
                    for entry, parts in zip(entries, restored):
                        reference = (tuple(entry.materialize_head(h) for h in range(entry.heads))
                                     if entry.high_offsets is not None else (entry.materialize(),))
                        for a, b in zip(parts, reference):
                            for x, y in zip(a, b):
                                torch.testing.assert_close(x, y, rtol=0, atol=0)

    @torch.inference_mode()
    def test_nonfinite_quantization_rejected_without_modifying_inputs(self):
        good = torch.randn(2, 17, 24, device='cuda')
        bad = good.clone()
        bad[1, 3, 4] = float('nan')
        entries = [LayerFrame.full(x, x, torch.float32) for x in (good, bad)]
        with self.assertRaisesRegex(ValueError, 'finite'):
            lower(entries, 8)
        torch.testing.assert_close(entries[0].high_k, good, rtol=0, atol=0)
        self.assertEqual(entries[0].bits, 16)

    @torch.inference_mode()
    def test_segmented_attention_matches_joined_prefix(self):
        for dtype in (torch.float16, torch.bfloat16):
            heads = []
            for h in range(4):
                parts = tuple((torch.randn(1, n, 64, device='cuda', dtype=dtype),
                               torch.randn(1, n, 64, device='cuda', dtype=dtype)) for n in (0, 3, 31+h*17, 137))
                heads.append(EncodingSegments(parts))
            q = torch.randn(1, 12, 196, 64, device='cuda', dtype=dtype)
            k = torch.randn(1, 4, 196, 64, device='cuda', dtype=dtype)
            v = torch.randn_like(k)
            expected = attend(q, k, v, [h.materialize() for h in heads])
            actual = attend(q, k, v, heads)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
