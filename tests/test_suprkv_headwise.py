"""Independent head masks, compact storage, quantization and ragged GQA decode."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from model.rekv.attention.rope import RotaryEmbeddingESM
from model.suprkv.attention import RetrieveLayer, streaming_attention_forward
from model.suprkv.cache import LayerFrame, StreamingKVCache
from model.suprkv.packed_attention import HeadwiseRotaryCache, PackedRotaryCache
from model.suprkv.quantization import CompactPackedKV, PackedKV
from model.suprkv.runtime import StreamingConfig
from model.suprkv.segments import summary_scores, segment_summary_scores, segment_frame_query_scores
from test_suprkv_rope import reference_rotate


def reference_attention(q, raw_k, values, rope, window, sink):
    """Dense FP32 oracle with explicit causal/window masks and partial RoPE."""
    total, count = raw_k.shape[-2], q.shape[-2]
    start = max(0, total-count-window)
    local = raw_k[:, :, start:]
    positions = torch.arange(local.shape[-2], device=q.device)
    qlocal = reference_rotate(q, positions[-count:], rope)
    klocal = reference_rotate(local, positions, rope).expand(-1, q.shape[1], -1, -1)
    scores = qlocal @ klocal.transpose(-1, -2) / q.shape[-1]**.5
    distance = positions[-count:, None] - positions[None, :]
    scores = scores.masked_fill((distance < 0) | (distance >= window), float('-inf'))
    payload = values[:, :, start:].expand(-1, q.shape[1], -1, -1)
    if total > window:
        qsink = reference_rotate(q, torch.full((count,), window-1, device=q.device), rope)
        ksink = raw_k[:, :, :sink].expand(-1, q.shape[1], -1, -1)
        sink_scores = qsink @ ksink.transpose(-1, -2) / q.shape[-1]**.5
        distance = torch.arange(total-count, total, device=q.device)[:, None] - torch.arange(sink, device=q.device)[None, :]
        sink_scores = sink_scores.masked_fill(distance < window, float('-inf'))
        scores = torch.cat((scores, sink_scores), -1)
        payload = torch.cat((payload, values[:, :, :sink].expand(-1, q.shape[1], -1, -1)), -2)
    return scores.softmax(-1) @ payload


class HeadwiseTests(unittest.TestCase):
    def test_default_restores_original_scoring_and_shared_cache_layout(self):
        torch.manual_seed(18)
        query = torch.randn(3, 6, 8)
        keys = list(torch.randn(3, 2, 5, 8))
        pooled = query.mean(0).reshape(2, 3, 8).mean(1)
        # Preserve the original contraction/reduction order, not just equality
        # to a post-hoc mean of the new headwise scores.
        reference = torch.einsum('hd,fhnd->fn', pooled, torch.stack(keys)) / (2*8**.5)
        torch.testing.assert_close(segment_summary_scores(query, keys), reference, rtol=0, atol=0)
        torch.testing.assert_close(summary_scores(query, keys[0]),
                                   torch.einsum('hd,hnd->n', pooled, keys[0])/(2*8**.5), rtol=0, atol=0)
        logits = torch.einsum('fhgd,hnd->hgfn', query.reshape(3, 2, 3, 8), torch.cat(keys, 1))/8**.5
        frame_reference = logits.mean(2).mean((0, 1)).reshape(3, 5)
        torch.testing.assert_close(segment_frame_query_scores(query, keys), frame_reference, rtol=0, atol=0)
        for strategy, expected in (('summary', reference), ('frame_mean_query', frame_reference)):
            for mode in ('layer', 'head'):
                kwargs = {} if mode == 'layer' else dict(compression_granularity='head')
                cache = StreamingKVCache([(keys[0][:, :2], keys[0][:, :2])], 100000, 5,
                                         dtype=torch.float32, group_size=8, **kwargs)
                for i, key in enumerate(keys):
                    cache.append_frame(i, [(key, key+1)], score_keys=[key], frame_queries=[query[i]])
                cache.close_segment([query] if strategy == 'summary' else None, scoring_strategy=strategy)
                for i, frame in cache.frames.items():
                    entry = frame.layers[0]
                    if mode == 'layer':
                        ids = torch.where(expected[i] >= expected[i].mean())[0]
                        self.assertIsNone(entry.high_offsets)
                        self.assertEqual(entry.high_indices.tolist(), ids.tolist())
                        torch.testing.assert_close(entry.high_k, keys[i][:, ids], rtol=0, atol=0)
                        packed = entry.drop_low().shrink(8)
                        self.assertIsInstance(packed.high_k, PackedKV)
                        original = PackedKV.quantize(keys[i][:, ids], 8, axis='channel', bits=8)
                        torch.testing.assert_close(packed.high_k.data, original.data, rtol=0, atol=0)
                    else:
                        self.assertIsNotNone(entry.high_offsets)
                        self.assertEqual(entry.high_k.ndim, 2)
                self.assertEqual(cache.memory_stats()['compression_granularity'], mode)
                cache.audit_accounting()

    def test_switch_defaults_and_validation(self):
        self.assertEqual(StreamingConfig().compression_granularity, 'layer')
        self.assertEqual(StreamingConfig(compression_granularity='head').compression_granularity, 'head')
        sink = torch.zeros(1, 2, 8)
        for invalid in ('heads', '', True, None):
            with self.assertRaisesRegex(ValueError, 'compression_granularity'):
                StreamingConfig(compression_granularity=invalid)
            with self.assertRaisesRegex(ValueError, 'compression_granularity'):
                StreamingKVCache([(sink, sink)], 100000, 5, compression_granularity=invalid)

    def test_each_head_uses_own_mean_and_owns_compact_storage(self):
        # Global averaging would erase both heads' distinct choices.
        scores = torch.tensor([[9., -1., -1., -1., -1.], [-9., 1., 1., 1., 1.]])
        key = torch.arange(2*5*8).reshape(2, 5, 8).float()
        source = LayerFrame.full(key, key+100, torch.float32)
        split = source.classify_scores(scores)
        self.assertEqual(split.high_offsets, (0, 1, 5))
        self.assertEqual(split.low_offsets, (0, 4, 5))
        self.assertEqual(split.high_indices.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(split.low_indices.tolist(), [1, 2, 3, 4, 0])
        pruned = split.drop_low()
        self.assertEqual(pruned.high_k.shape, (5, 8))
        self.assertEqual(pruned.high_k.untyped_storage().nbytes(), 5*8*4)
        self.assertNotEqual(pruned.high_k.data_ptr(), source.high_k.data_ptr())
        for h, ids in enumerate(([0], [1, 2, 3, 4])):
            k, v, indices = pruned.materialize_head(h)
            self.assertEqual(indices.tolist(), ids)
            torch.testing.assert_close(k[0], key[h, ids])
            torch.testing.assert_close(v[0], key[h, ids]+100)
        full_k, full_v, full_indices = split.materialize()
        torch.testing.assert_close(full_k, key)
        torch.testing.assert_close(full_v, key+100)
        self.assertEqual(full_indices.tolist(), list(range(5)))
        with self.assertRaisesRegex(ValueError, 'Unequal head lengths'):
            pruned.materialize()

    def test_ties_are_high_and_heads_with_no_low_tokens_are_valid(self):
        key = torch.randn(3, 5, 8)
        layer = LayerFrame.full(key, key).classify_scores(torch.tensor([
            [0., 0., 0., 0., 0.], [-2., 0., 0., 0., 2.], [1., 1., 1., 1., 1.]]))
        self.assertEqual(layer.high_offsets, (0, 5, 9, 14))
        self.assertEqual(layer.low_offsets, (0, 0, 1, 1))
        layer = layer.drop_low().shrink(8)
        for h, count in enumerate((5, 4, 5)):
            self.assertEqual(layer.materialize_head(h)[0].shape, (1, count, 8))
        self.assertEqual(layer.low_offsets, (0, 0, 0, 0))

    def test_budget_prunes_independently_and_preserves_recent_and_snapshot(self):
        sink = torch.zeros(2, 2, 128)
        key = torch.tensor([[9., -1., -1., -1., -1.], [-9., 1., 1., 1., 1.]])
        key = key[..., None].expand(2, 5, 128)
        cache = StreamingKVCache([(sink, sink)], 100000, 5, recent_frames=1, compression_granularity="head")
        for i in range(3):
            cache.append_frame(i, [(key, key+10)], score_keys=[key])
        cache.close_segment([torch.ones(2, 4, 128)])
        self.assertEqual(cache.counts['scoring_failures'], 0)
        for frame in cache.frames.values():
            self.assertEqual(frame.layers[0].high_offsets, (0, 1, 5))
        snapshot = cache.acquire_snapshot(1)
        protected = cache.frames[2]
        cache.budget = cache.persistent_bytes()-100
        cache._enforce_budget()
        self.assertEqual(cache.counts['dropped_low_frames'], 2)
        self.assertEqual(cache.counts['evicted_frames'], 0)
        self.assertLessEqual(cache.persistent_bytes(), cache.budget)
        self.assertIs(cache.frames[2], protected)
        for i in (0, 1):
            layer = cache.frames[i].layers[0]
            self.assertEqual(layer.bits, 16)
            self.assertEqual(layer.high_offsets, (0, 1, 5))
            self.assertEqual(layer.low_offsets, (0, 0, 0))
            self.assertEqual(snapshot.frames[i].layers[0].low_offsets, (0, 4, 5))
        cache.audit_accounting()
        cache.release_snapshot(1)
        cache.audit_accounting()

    def test_compact_quantization_matches_independent_heads_at_every_precision(self):
        torch.manual_seed(101)
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                for axis in ('channel', 'token'):
                    # Distinct magnitudes expose accidental groups spanning heads.
                    lengths = (1, 9, 37)
                    offsets = (0, 1, 10, 47)
                    x = torch.cat([torch.randn(n, 10, device=device)*(.1+100*h)
                                   for h, n in enumerate(lengths)]).to(dtype)
                    compact = CompactPackedKV.quantize(x, offsets, 24, axis=axis)
                    reference = [PackedKV.quantize(x[a:b][None], 24, axis=axis, bits=8, backend='torch')
                                 for a, b in zip(offsets, offsets[1:])]
                    for bits in (8, 4, 2):
                        self.assertEqual(compact.bits, bits)
                        self.assertEqual(compact.data.numel(), sum(p.data.numel() for p in reference))
                        for h, expected in enumerate(reference):
                            actual = compact.head(h)
                            for field in ('data', 'scale', 'offset'):
                                torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
                            torch.testing.assert_close(actual.dequantize(dtype=dtype),
                                                       expected.dequantize(dtype=dtype, backend='torch'), rtol=0, atol=0)
                        torch.testing.assert_close(compact.dequantize(dtype=dtype),
                                                   torch.cat([p.dequantize(dtype=dtype)[0] for p in reference]), rtol=0, atol=0)
                        compact = compact.shrink()
                        reference = [p.shrink(backend='torch') for p in reference]

    def make_cache(self, bits, device='cpu', dtype=torch.float32):
        torch.manual_seed(47)
        rope = RotaryEmbeddingESM(8, distance_scale=.7).to(device)
        sink = torch.randn(2, 2, 16, device=device, dtype=dtype)
        cache = StreamingKVCache([(sink, torch.randn_like(sink))], 100000, 5,
                                 recent_frames=1, group_size=8, dtype=dtype, compression_granularity="head")
        for i in range(3):
            raw = torch.randn(2, 5, 16, device=device, dtype=dtype)
            positions = torch.arange(17+i*5, 22+i*5, device=device)
            key = reference_rotate(raw, positions, rope)
            cache.append_frame(i, [(key, torch.randn_like(key))], score_keys=[raw], rope_start=17+i*5)
        cache.close_segment([torch.zeros(2, 4, 16, device=device)])
        # Distinct counts AND different positions in the two heads. The recent
        # frame remains complete, as in actual budget enforcement.
        scores = torch.tensor([[9., -1., -1., -1., -1.], [-9., 1., 1., 1., 1.]], device=device)
        for i in range(2):
            old = cache.frames[i]
            k, v, _ = old.layers[0].materialize()
            layer = LayerFrame.full(k, v, dtype).classify_scores(scores).drop_low()
            while layer.bits > bits:
                layer = layer.shrink(8)
            cache.frames[i] = replace(old, layers=(layer,))
        return cache, rope

    def forward(self, q, k, v, past, rope, window, packed=False):
        flat = lambda x: x.transpose(1, 2).flatten(-2)
        fn = streaming_attention_forward(window, 2, fattn=False, packed_attention=packed)
        out, past = fn(SimpleNamespace(), flat(q), flat(k), rope, True, past,
                       lambda _: flat(q), lambda _: flat(k), lambda _: flat(v), lambda x: x,
                       q.shape[-1], q.shape[1], k.shape[1])
        return out.view(1, q.shape[-2], q.shape[1], q.shape[-1]).transpose(1, 2), past

    @torch.inference_mode()
    def test_gqa_prefill_decode_and_window_rollover_match_independent_dense_oracle(self):
        for bits in (16, 8, 4, 2):
            cache, rope = self.make_cache(bits)
            snapshot = cache.acquire_snapshot(1)
            retrieved, ids = snapshot.retrieve_layer(0, torch.ones(1, 4, 1, 16), 3, headwise=True)
            self.assertEqual(ids, (0, 1, 2))
            self.assertEqual([h.key.shape[-2] for h in retrieved.heads], [9, 15])
            with self.assertRaisesRegex(ValueError, 'headwise=True'):
                snapshot.retrieve_layer(0, torch.ones(1, 4, 1, 16), 3)
            raw = [reference_rotate(h.key, -h.positions, rope) for h in retrieved.heads]
            values = [h.value for h in retrieved.heads]
            hits = set()
            past = RetrieveLayer(snapshot, 0, 3, hits)
            for step, (length, window) in enumerate(((3, 100), (2, 100), (1, 4), (3, 4), (1, 4))):
                q, k = torch.randn(1, 4, length, 16), torch.randn(1, 2, length, 16)
                v = torch.randn_like(k)
                expected = []
                for h in range(2):
                    all_k, all_v = torch.cat((raw[h], k[:, h:h+1]), -2), torch.cat((values[h], v[:, h:h+1]), -2)
                    expected.append(reference_attention(q[:, h*2:h*2+2], all_k, all_v, rope, window, 2))
                    if step:  # Retrieval probe is not part of the generated prefix.
                        raw[h], values[h] = all_k, all_v
                        if all_k.shape[-2] > window+2:
                            raw[h] = torch.cat((all_k[:, :, :2], all_k[:, :, -window:]), -2)
                            values[h] = torch.cat((all_v[:, :, :2], all_v[:, :, -window:]), -2)
                actual, past = self.forward(q, k, v, past, rope, window)
                self.assertIsInstance(past, HeadwiseRotaryCache)
                torch.testing.assert_close(actual, torch.cat(expected, 1), rtol=3e-5, atol=3e-6)
            self.assertEqual(hits, {0, 1, 2})
            cache.audit_accounting()
            cache.release_snapshot(1)

    def test_packed_views_rebase_each_head_and_accounting_keeps_snapshots_immutable(self):
        cache, rope = self.make_cache(4)
        snapshot = cache.acquire_snapshot(1)
        query = torch.ones(1, 4, 1, 16)
        dense, _ = snapshot.retrieve_layer(0, query, 3, headwise=True)
        packed, _ = snapshot.retrieve_layer(0, query, 3, headwise=True, packed=True, position_bias=rope)
        for raw, compressed in zip(dense.heads, packed.heads):
            k, v, ranks = compressed.materialize()
            torch.testing.assert_close(k, reference_rotate(raw.key, ranks-raw.positions, rope), rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(v, raw.value)
        before = snapshot.frames[0].layers[0].high_k.data.clone()
        old = cache.frames[0]
        cache.frames[0] = replace(old, layers=(old.layers[0].shrink(8),))
        cache.audit_accounting()
        torch.testing.assert_close(snapshot.frames[0].layers[0].high_k.data, before, rtol=0, atol=0)
        state = cache.frame_states()
        self.assertEqual(state['count_unit'], 'mean_tokens_per_kv_head')
        self.assertEqual(state['frames'][0]['head_counts'], [[[0, 0, 0, 1], [0, 0, 0, 4]]])
        self.assertEqual(state['frames'][0]['counts'], [[0, 0, 0, 2.5]])
        cache.release_snapshot(1)
        cache.audit_accounting()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    @torch.inference_mode()
    def test_cuda_packed_ragged_attention_without_dense_materialization(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for bits in (8, 4, 2):
                cache, rope = self.make_cache(bits, 'cuda', dtype)
                snapshot = cache.acquire_snapshot(1)
                dense, packed = RetrieveLayer(snapshot, 0, 3, set()), RetrieveLayer(snapshot, 0, 3, set())
                for step, (length, window) in enumerate(((3, 100), (1, 100), (2, 4), (1, 4))):
                    q = torch.randn(1, 4, length, 16, device='cuda', dtype=dtype)
                    k, v = torch.randn_like(q[:, :2]), torch.randn_like(q[:, :2])
                    expected, dense = self.forward(q, k, v, dense, rope, window)
                    if step < 2:
                        with patch.object(PackedKV, 'dequantize', side_effect=AssertionError('dense KV')), \
                             patch.object(LayerFrame, 'materialize_head', side_effect=AssertionError('dense frame')):
                            actual, packed = self.forward(q, k, v, packed, rope, window, True)
                        self.assertTrue(all(isinstance(h, PackedRotaryCache) for h in packed.heads))
                    else:
                        actual, packed = self.forward(q, k, v, packed, rope, window, True)
                    tol = .03 if dtype == torch.bfloat16 else .006 if dtype == torch.float16 else 3e-5
                    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


if __name__ == '__main__':
    unittest.main()
