"""Numerical and ownership checks for batched headwise CUDA inference."""
from dataclasses import replace
from unittest.mock import patch
import unittest

import torch

import test_suprkv_headwise as previous
from model.rekv.attention.rope import RotaryEmbeddingESM
from model.suprkv.attention import RetrieveLayer
from model.suprkv.cache import LayerFrame, StreamingKVCache
from model.suprkv.ragged_cache import RaggedKVCache
from model.suprkv.rotary import rotate_at


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class RaggedTests(unittest.TestCase):
    @torch.inference_mode()
    def test_full_rotary_large_positions_and_mixed_frame_formats(self):
        for dtype in (torch.float16, torch.float32):
            rope = RotaryEmbeddingESM(128, distance_scale=.7).cuda()
            sink = torch.randn(4, 16, 128, dtype=dtype, device='cuda')
            cache = StreamingKVCache([(sink, sink)], 100000, 196, dtype=dtype,
                                     compression_granularity='head')
            for i in range(3):
                raw = torch.randn(4, 196, 128, dtype=dtype, device='cuda')
                key = rotate_at(raw, torch.arange(10000+i*196, 10196+i*196, device='cuda'), rope)
                cache.append_frame(i, [(key, raw)], rope_start=10000+i*196)
                old = cache.frames[i]
                entry = old.layers[0]
                if i < 2:
                    entry = entry.classify_scores(torch.randn(4, 196, device='cuda')).shrink(128)
                    if i == 0:
                        entry = entry.drop_low().shrink(128)
                cache.frames[i] = replace(old, layers=(entry,), closed=True)
            snapshot = cache.acquire_snapshot(1)
            q = torch.randn(1, 28, 1, 128, device='cuda', dtype=dtype)
            old, _ = snapshot.retrieve_layer(0, q, 3, position_bias=rope, headwise=True)
            new, _ = snapshot.retrieve_layer(0, q, 3, position_bias=rope, batched=True)
            for head, start, n in zip(old.heads, new.host_offsets, new.host_lengths):
                expected = rotate_at(head.key, torch.arange(n, device='cuda')-head.positions, rope)
                tol = .004 if dtype == torch.float16 else 3e-5
                torch.testing.assert_close(new.key[start:start+n], expected[0, 0], rtol=tol, atol=tol)
                torch.testing.assert_close(new.value[start:start+n], head.value[0, 0], rtol=0, atol=0)

    @torch.inference_mode()
    def test_restore_all_bits_both_partitions_and_probe_does_not_commit(self):
        helper = previous.HeadwiseTests()
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for bits in (16, 8, 4, 2):
                cache, rope = helper.make_cache(bits, 'cuda', dtype)
                # Mix a quantized high partition with a present low partition.
                old = cache.frames[1]
                key = torch.randn(2, 5, 16, dtype=dtype, device='cuda')
                entry = LayerFrame.full(key, torch.randn_like(key), dtype).classify_scores(
                    torch.tensor([[1., -2., 4., -1., 5.], [-3., 9., -2., 7., -1.]], device='cuda'))
                while entry.bits > bits:
                    entry = entry.shrink(8)
                cache.frames[1] = replace(old, layers=(entry,))
                snapshot = cache.acquire_snapshot(1)
                query = torch.randn(1, 4, 3, 16, dtype=dtype, device='cuda')
                reference, ids = snapshot.retrieve_layer(0, query, 3, headwise=True, position_bias=rope)
                with patch.object(LayerFrame, 'materialize_head', side_effect=AssertionError('per-head restore')):
                    actual, actual_ids = snapshot.retrieve_layer(0, query, 3, batched=True, position_bias=rope)
                self.assertEqual(ids, actual_ids)
                self.assertEqual(actual.host_lengths, tuple(x.key.shape[-2] for x in reference.heads))
                tol = .025 if dtype == torch.bfloat16 else .004 if dtype == torch.float16 else 3e-5
                for head, start, n in zip(reference.heads, actual.host_offsets, actual.host_lengths):
                    expected = rotate_at(head.key, torch.arange(n, device='cuda')-head.positions, rope)
                    torch.testing.assert_close(actual.key[start:start+n], expected[0, 0], rtol=tol, atol=tol)
                    torch.testing.assert_close(actual.value[start:start+n], head.value[0, 0], rtol=0, atol=0)
                keys = actual.key.clone()
                q, k, v = query, query[:, :2].contiguous(), torch.randn_like(query[:, :2]).contiguous()
                actual.attend(q, k, v, rope, commit=False)
                self.assertEqual(actual.generated, 0)
                for start, n in zip(actual.host_offsets, actual.host_lengths):
                    torch.testing.assert_close(actual.key[start:start+n], keys[start:start+n], rtol=0, atol=0)
                cache.release_snapshot(1)
                cache.audit_accounting()

    @torch.inference_mode()
    def test_prefill_decode_and_window_fallback_match_legacy(self):
        helper = previous.HeadwiseTests()
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            cache, rope = helper.make_cache(4, 'cuda', dtype)
            snapshot = cache.acquire_snapshot(1)
            fast, legacy = RetrieveLayer(snapshot, 0, 3, set()), RetrieveLayer(snapshot, 0, 3, set())
            for step, (length, window) in enumerate(((3, 100), (5, 100), (1, 100), (1, 100), (3, 4), (1, 4))):
                q = torch.randn(1, 4, length, 16, device='cuda', dtype=dtype)
                k, v = torch.randn_like(q[:, :2]), torch.randn_like(q[:, :2])
                with patch('model.suprkv.attention.ragged_supported', return_value=False):
                    expected, legacy = helper.forward(q, k, v, legacy, rope, window)
                actual, fast = helper.forward(q, k, v, fast, rope, window)
                if step < 4:
                    self.assertIsInstance(fast, RaggedKVCache)
                tol = .03 if dtype == torch.bfloat16 else .006 if dtype == torch.float16 else 4e-5
                torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)

    @torch.inference_mode()
    def test_long_gqa_attention_matches_sdpa_and_reuses_buffers(self):
        torch.manual_seed(41)
        for dim in (8, 16, 80, 128):
            lengths = (1, 513, 1103, 2049)
            rope = RotaryEmbeddingESM(dim if dim == 128 else dim//2, distance_scale=.7).cuda()
            cache = RaggedKVCache.allocate(lengths, dim, torch.float16, 'cuda', reserve=4)
            raw, values = [], []
            for n, start in zip(lengths, cache.host_offsets):
                k = torch.randn(n, dim, device='cuda', dtype=torch.float16)
                v = torch.randn_like(k)
                raw.append(k); values.append(v)
                cache.key[start:start+n].copy_(rotate_at(k, torch.arange(n, device='cuda'), rope))
                cache.value[start:start+n].copy_(v)
            for length in (3, 1, 1, 19, 1):
                query_heads = 28 if dim == 128 else 8
                group = query_heads//4
                q = torch.randn(1, query_heads, length, dim, device='cuda', dtype=torch.float16)
                k, v = torch.randn_like(q[:, :4]), torch.randn_like(q[:, :4])
                before, growth = cache.key.data_ptr(), cache.generated+length>cache.host_offsets[1]-lengths[0]
                result = cache.attend(q, k, v, rope, commit=True)
                if not growth:
                    self.assertEqual(cache.key.data_ptr(), before)
                expected = []
                for h in range(4):
                    old = len(raw[h])
                    raw[h] = torch.cat((raw[h], k[0, h]), 0)
                    values[h] = torch.cat((values[h], v[0, h]), 0)
                    qr = rotate_at(q[:, group*h:group*(h+1)], torch.arange(old, old+length, device='cuda'), rope)
                    kr = rotate_at(raw[h][None, None], torch.arange(old+length, device='cuda'), rope)
                    mask = torch.arange(old+length, device='cuda')[None] <= torch.arange(old, old+length, device='cuda')[:, None]
                    expected.append(torch.nn.functional.scaled_dot_product_attention(qr, kr, values[h][None, None], attn_mask=mask))
                torch.testing.assert_close(result, torch.cat(expected, 1), rtol=.006, atol=.006)


if __name__ == '__main__':
    unittest.main()
