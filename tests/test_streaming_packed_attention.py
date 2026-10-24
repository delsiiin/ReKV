"""Numerical and integration checks for direct packed-cache attention."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch

from model.attention.rope import RotaryEmbeddingESM
from model.streaming.attention import RetrieveLayer, RotaryCache, streaming_attention_forward
from model.streaming.cache import CacheSnapshot, Frame, LayerFrame
from model.streaming.packed_attention import PackedRotaryCache
from model.streaming.quantization import PackedKV


class PackedChannelTests(unittest.TestCase):
    def test_both_axes_gqa_and_autograd_reference(self):
        for axis in ('channel', 'token'):
            packed = PackedKV.quantize(torch.randn(2, 11, 10), 8, axis=axis, bits=4)
            for transpose in (True, False):
                left = torch.randn(4, 3, 10 if transpose else 11, requires_grad=True)
                result = packed.matmul(left, transpose=transpose)
                dense = packed.dequantize(dtype=left.dtype).repeat_interleave(2, 0)
                ref = left @ (dense.transpose(-1, -2) if transpose else dense)
                torch.testing.assert_close(result, ref)
                result.sum().backward()
                self.assertIsNotNone(left.grad)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class PackedAttentionTests(unittest.TestCase):
    def forward(self, q, k, v, past, rope, window, packed):
        flat = lambda x: x.transpose(1, 2).flatten(-2)
        fn = streaming_attention_forward(window, 2, fattn=q.dtype != torch.float32,
                                         packed_attention=packed)
        result, past = fn(SimpleNamespace(), flat(q), flat(k), rope, True, past,
                          lambda _: flat(q), lambda _: flat(k), lambda _: flat(v), lambda x: x,
                          q.shape[-1], q.shape[1], k.shape[1])
        return result, past

    def snapshot(self, dtype, bits):
        frames = []
        for i in range(2):
            k = torch.randn(2, 11, 16, device='cuda', dtype=dtype)
            entry = LayerFrame.full(k, torch.randn_like(k), dtype)
            entry = entry.classify_scores((torch.arange(11, device='cuda') % 2).float())
            if i == 1:
                entry = entry.drop_low()
            frame_bits = bits if i == 0 else {16: 8, 8: 4, 4: 2, 2: 16}[bits]
            entry = replace(entry, high_k=PackedKV.quantize(entry.high_k, 8, axis='channel', bits=frame_bits),
                            high_v=PackedKV.quantize(entry.high_v, 8, bits=frame_bits), bits=frame_bits)
            frames.append(Frame(i, float(i), 0, (entry,), True, rope_start=31+i*11, token_count=11))
        sink = torch.randn(2, 2, 16, device='cuda', dtype=dtype)
        return CacheSnapshot(0, 1, ((sink, torch.randn_like(sink)),), tuple(frames), (1,))

    @torch.inference_mode()
    def test_prefill_decode_no_materialization_and_window_fallback(self):
        torch.manual_seed(53)
        rope = RotaryEmbeddingESM(8, distance_scale=.7).cuda()  # Partial RoPE.
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for bits in (16, 8, 4, 2):
                snapshot = self.snapshot(dtype, bits)
                dense_hits, packed_hits = set(), set()
                dense_past = RetrieveLayer(snapshot, 0, 2, dense_hits)
                packed_past = RetrieveLayer(snapshot, 0, 2, packed_hits)
                for step, length in enumerate((3, 1, 2, 1)):
                    q = torch.randn(1, 4, length, 16, device='cuda', dtype=dtype)
                    k, v = torch.randn_like(q[:, :2]), torch.randn_like(q[:, :2])
                    # Last step deliberately rolls over into the sliding/sink path.
                    window = 100 if step < 3 else 4
                    expected, dense_past = self.forward(q, k, v, dense_past, rope, window, False)
                    if step < 3:
                        with patch.object(PackedKV, 'dequantize', side_effect=AssertionError('dense decode')), \
                             patch.object(LayerFrame, 'materialize', side_effect=AssertionError('dense frame')):
                            actual, packed_past = self.forward(q, k, v, packed_past, rope, window, True)
                        self.assertIsInstance(packed_past, PackedRotaryCache)
                        self.assertEqual(packed_past.length, dense_past.key.shape[-2])
                    else:
                        actual, packed_past = self.forward(q, k, v, packed_past, rope, window, True)
                        self.assertIsInstance(packed_past, RotaryCache)
                    tol = .03 if dtype == torch.bfloat16 else .006 if dtype == torch.float16 else 3.e-5
                    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
                self.assertEqual(dense_hits, packed_hits)
                self.assertEqual(packed_hits, {0, 1})

    @torch.inference_mode()
    def test_channel_codes_tail_groups_and_nondefault_stream(self):
        with torch.cuda.stream(torch.cuda.Stream()):
            for group in (8, 24, 128):
                x = torch.randn(2, 37, 10, device='cuda')
                actual = PackedKV.quantize(x, group, axis='channel', backend='triton')
                ref = PackedKV.quantize(x, group, axis='channel', backend='torch')
                for bits in (16, 8, 4, 2):
                    torch.testing.assert_close(actual.data, ref.data, rtol=0, atol=0)
                    torch.testing.assert_close(actual.dequantize(dtype=torch.float32),
                                               ref.dequantize(dtype=torch.float32, backend='torch'), rtol=0, atol=0)
                    for transpose in (False, True):
                        a = torch.randn(4, 1, 10 if transpose else 37, device='cuda')
                        torch.testing.assert_close(actual.matmul(a, transpose=transpose),
                                                   ref.matmul(a, transpose=transpose, backend='torch'), rtol=3.e-5, atol=3.e-5)
                    actual, ref = actual.shrink(), ref.shrink(backend='torch')
        torch.cuda.synchronize()

    @torch.inference_mode()
    def test_split_k_real_head_sizes_and_all_masked_padding(self):
        from model.streaming.packed_attention import AttentionChunk, coalesce
        for dim in (128, 256):
            rope = RotaryEmbeddingESM(dim).cuda()
            chunks = []
            start = 0
            group = 128 if dim == 128 else 512
            for length in (257, 17):
                k = PackedKV.quantize(torch.randn(2, length, dim, device='cuda'), group, axis='channel', bits=4)
                v = PackedKV.quantize(torch.randn(2, length, dim, device='cuda'), group, bits=4)
                ranks = torch.arange(start, start+length, device='cuda')
                chunks.append(AttentionChunk.create(k, v, ranks, ranks+17, rope))
                start += length
            cache = PackedRotaryCache(coalesce(chunks), start, torch.float16)
            q = torch.randn(1, 4, 3, dim, device='cuda', dtype=torch.float16)
            k, v = torch.randn_like(q[:, :2]), torch.randn_like(q[:, :2])
            rk, rv, rp = cache.materialize()
            dense = RotaryCache(rk, rv, rp)
            expected, _ = self.forward(q, k, v, dense, rope, 1000, False)
            actual, _ = self.forward(q, k, v, cache, rope, 1000, True)
            torch.testing.assert_close(actual, expected, rtol=.006, atol=.006)
            # Split-K GEMV reduction, including an incomplete final K tile.
            a = torch.randn(4, 1, 257, device='cuda', dtype=torch.float16)
            for bits in (8, 4, 2):
                p = PackedKV.quantize(torch.randn(2, 257, dim, device='cuda'), 128, bits=bits)
                torch.testing.assert_close(p.matmul(a), p.matmul(a, backend='torch'), rtol=.003, atol=.003)

    @torch.inference_mode()
    def test_qwen_model_bridge_retains_packed_cache(self):
        from transformers import Qwen2Config, Qwen2Model
        from model.patch import patch_hf
        model = Qwen2Model(Qwen2Config(vocab_size=32, hidden_size=64, intermediate_size=96,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)).half().cuda().eval()
        snapshot = self.snapshot(torch.float16, 4)
        patch_hf(model, attention_factory=streaming_attention_forward, n_local=100, n_init=2,
                 fattn=True, packed_attention=False)
        tokens = torch.tensor([[1, 2, 3]], device='cuda')
        reference = model(input_ids=tokens, past_key_values=[RetrieveLayer(snapshot, 0, 2, set())], use_cache=True)
        expected = model(input_ids=tokens[:, :1], past_key_values=reference.past_key_values, use_cache=True)
        patch_hf(model, attention_factory=streaming_attention_forward, n_local=100, n_init=2,
                 fattn=True, packed_attention=True)
        with patch.object(PackedKV, 'dequantize', side_effect=AssertionError('dense KV')):
            actual = model(input_ids=tokens, past_key_values=[RetrieveLayer(snapshot, 0, 2, set())], use_cache=True)
            decoded = model(input_ids=tokens[:, :1], past_key_values=actual.past_key_values, use_cache=True)
        self.assertIsInstance(actual.past_key_values[0], PackedRotaryCache)
        torch.testing.assert_close(actual.last_hidden_state, reference.last_hidden_state, rtol=.006, atol=.006)
        torch.testing.assert_close(decoded.last_hidden_state, expected.last_hidden_state, rtol=.006, atol=.006)


if __name__ == '__main__':
    unittest.main()
