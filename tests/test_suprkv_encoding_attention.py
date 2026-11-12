"""Head-batched full-prefix encoding: ragged lengths, causality and RoPE."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from model.rekv.attention.rope import RotaryEmbeddingESM
from model.rekv.attention.dot_production_attention import get_multi_stage_dot_production_attention
from model.suprkv.attention import EncodingKV, EncodeLayer, streaming_attention_forward
from model.suprkv.encoding_attention import attend, supported
from model.suprkv.packed_attention import HeadwiseRotaryCache


def legacy(query, key, value, heads):
    cls, _ = get_multi_stage_dot_production_attention(True)
    group = query.shape[1] // len(heads)
    outputs = []
    for h, past in enumerate(heads):
        q = query[:, h*group:(h+1)*group]
        k = torch.cat((past.key, key[:, h:h+1]), -2)
        v = torch.cat((past.value, value[:, h:h+1]), -2)
        attention = cls(q.shape, q.dtype, q.device)
        attention.append(q, k, v, sliding_window=k.shape[-2], end=True)
        outputs.append(attention.get_result()[0])
    return torch.cat(outputs, 1)


def inputs(lengths, count, dim, group, dtype):
    # Noncontiguous prefixes also exercise the descriptor's token/channel strides.
    heads = tuple(EncodingKV(torch.randn(1, 1, n*2, dim*2, device='cuda', dtype=dtype)[..., ::2, ::2],
                             torch.randn(1, 1, n*2, dim*2, device='cuda', dtype=dtype)[..., ::2, ::2])
                  for n in lengths)
    q = torch.randn(1, len(heads)*group, count, dim, device='cuda', dtype=dtype)
    k = torch.randn(1, len(heads), count, dim, device='cuda', dtype=dtype)
    return q, k, torch.randn_like(k), heads


class DispatchTests(unittest.TestCase):
    def test_cpu_is_not_dispatched_to_cuda(self):
        self.assertFalse(supported(torch.empty(1, 4, 196, 128)))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class EncodingAttentionCudaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(351)
        torch.set_num_threads(1)

    @torch.inference_mode()
    def test_ragged_gqa_matches_per_head_and_sdpa(self):
        cases = (((0, 1, 31, 65), 7, 16, 2), ((63, 128), 1, 64, 1),
                 ((1, 513, 1103, 2049), 196, 128, 7), ((32, 64), 64, 32, 3))
        for dtype in (torch.float16, torch.bfloat16):
            for lengths, count, dim, group in cases:
                with self.subTest(dtype=dtype, lengths=lengths, count=count, dim=dim):
                    q, k, v, heads = inputs(lengths, count, dim, group, dtype)
                    expected = legacy(q, k, v, heads)
                    actual = attend(q, k, v, heads)
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    for h, past in enumerate(heads):
                        keys = torch.cat((past.key, k[:, h:h+1]), -2).float().repeat_interleave(group, 1)
                        values = torch.cat((past.value, v[:, h:h+1]), -2).float().repeat_interleave(group, 1)
                        mask = (torch.arange(keys.shape[-2], device='cuda')[None] <=
                                lengths[h]+torch.arange(count, device='cuda')[:, None])
                        reference = F.scaled_dot_product_attention(q[:, h*group:(h+1)*group].float(), keys, values, attn_mask=mask)
                        tolerance = 0.015 if dtype == torch.bfloat16 else 0.002
                        torch.testing.assert_close(actual[:, h*group:(h+1)*group].float(), reference,
                                                   rtol=tolerance, atol=tolerance)

    @torch.inference_mode()
    def test_current_frame_is_causal_and_history_is_immutable(self):
        q, k, v, heads = inputs((0, 63, 129, 513), 7, 64, 2, torch.bfloat16)
        before = [(h.key.clone(), h.value.clone()) for h in heads]
        original = attend(q, k, v, heads)
        k[..., 4:, :] *= 10
        v[..., 4:, :] += 100
        changed = attend(q, k, v, heads)
        torch.testing.assert_close(original[..., :4, :], changed[..., :4, :], rtol=0, atol=0)
        for head, (pk, pv) in zip(heads, before):
            torch.testing.assert_close(head.key, pk, rtol=0, atol=0)
            torch.testing.assert_close(head.value, pv, rtol=0, atol=0)

    @torch.inference_mode()
    def test_encoding_dispatch_preserves_absolute_rope_and_score_capture(self):
        q, k, v, heads = inputs((31, 65), 7, 64, 3, torch.bfloat16)
        rope = RotaryEmbeddingESM(64).cuda()
        flatten = lambda t: t.transpose(1, 2).flatten(-2)
        fn = streaming_attention_forward(4, 2, fattn=True)
        results, contexts = [], []
        for batched in (False, True):
            context = EncodeLayer(HeadwiseRotaryCache(heads), next_position=100000,
                                  capture_scores=True, capture_frame_query=True)
            with patch('model.suprkv.attention.encoding_attention_supported', return_value=batched):
                out, returned = fn(SimpleNamespace(), flatten(q), flatten(k), rope, True, context,
                    lambda _: flatten(q), lambda _: flatten(k), lambda _: flatten(v), lambda t: t, 64, 6, 2)
            self.assertIs(returned, context)
            self.assertEqual(context.position_start, 100000)
            results.append(out)
            contexts.append(context)
        torch.testing.assert_close(*results, rtol=0, atol=0)
        for attr in ('score_key', 'frame_query'):
            torch.testing.assert_close(getattr(contexts[0], attr), getattr(contexts[1], attr), rtol=0, atol=0)
        for a, b in zip(contexts[0].captured, contexts[1].captured):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
