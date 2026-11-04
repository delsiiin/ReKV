"""Compact quantization and frame Query reduction against torch oracles.

TRITON_INTERPRET=1 runs kernels on CPU. Triton's interpreter cannot execute
libdevice externs: only nearbyint is emulated with NumPy's nearest-even rint.
CUDA runs use the real libdevice function without that substitution.
"""
from contextlib import ExitStack
import importlib.util
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from model.rekv.attention.rope import RotaryEmbeddingESM
from model.suprkv.attention import EncodeLayer, RotaryCache, streaming_attention_forward
from model.suprkv.cache import StreamingKVCache
from model.suprkv.quantization import CompactPackedKV, PackedKV, _compact_quantization_supported
from model.suprkv.segments import frame_query_mean, _fused_frame_query_mean_supported


INTERPRETER = os.environ.get('TRITON_INTERPRET') == '1'


def interpreter_nearbyint(x):
    import numpy as np
    import triton.language as tl
    from triton.runtime.interpreter import TensorHandle
    return tl.tensor(TensorHandle(np.rint(x.handle.data), x.dtype.scalar), x.type)


class P1FallbackTests(unittest.TestCase):
    def test_cpu_query_mean_and_validation(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            q = torch.randn(3, 7, 11, dtype=dtype, requires_grad=True)
            self.assertFalse(_fused_frame_query_mean_supported(q))
            mean = frame_query_mean(q)
            torch.testing.assert_close(mean, q.float().mean(1), rtol=0, atol=0)
            self.assertFalse(mean.requires_grad)
        for q in (torch.ones(2, 3), torch.empty(2, 0, 3), torch.ones(2, 3, 4, dtype=torch.int32)):
            with self.assertRaises(ValueError):
                frame_query_mean(q)

    def test_compact_options_offsets_and_nonfinite_validation(self):
        x = torch.randn(10, 7)
        self.assertFalse(_compact_quantization_supported(x, 8))
        for offsets in ((), (0,), (0, 0, 10), (0, 9, 8, 10), (1, 10), (0, 9), (0, 1.5, 10)):
            with self.assertRaises(ValueError):
                CompactPackedKV.quantize(x, offsets, 8, axis='channel')
        for extra in (dict(axis='bad'), dict(bits=3), dict(backend='bad'), dict(group_size=7),
                      dict(backend='triton')):
            options = dict(axis='channel', group_size=8)
            options.update(extra)
            with self.assertRaises(ValueError):
                CompactPackedKV.quantize(x, (0, 3, 10), **options)
        for bad in (float('nan'), float('inf')):
            value = x.clone()
            value[-1, -1] = bad
            with self.assertRaisesRegex(ValueError, 'finite'):
                CompactPackedKV.quantize(value, (0, 3, 10), 8, axis='token')


@unittest.skipUnless((torch.cuda.is_available() or INTERPRETER) and importlib.util.find_spec('triton'),
                     'CUDA or TRITON_INTERPRET=1 required')
class P1FusionTests(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu' if INTERPRETER else 'cuda'
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.context.enter_context(torch.inference_mode())
        if INTERPRETER:
            from model.suprkv import packed_kernels
            self.context.enter_context(patch.object(packed_kernels.libdevice, 'nearbyint', new=interpreter_nearbyint))
            for target in ('model.suprkv.quantization._compact_quantization_supported',
                           'model.suprkv.segments._fused_frame_query_mean_supported',
                           'model.suprkv.segments._fused_frame_scores_supported',
                           'model.suprkv.cache._fused_partition_supported'):
                self.context.enter_context(patch(target, return_value=True))

    def assert_packed_equal(self, actual, expected):
        for name in ('data', 'scale', 'offset'):
            x, y = getattr(actual, name), getattr(expected, name)
            torch.testing.assert_close(x, y, rtol=0, atol=0)
            self.assertEqual(x.untyped_storage().nbytes(), x.numel()*x.element_size())
        for name in ('head_offsets', 'data_offsets', 'parameter_offsets', 'dim', 'bits', 'group_size', 'axis'):
            self.assertEqual(getattr(actual, name), getattr(expected, name))

    def test_compact_exact_codes_parameters_heads_tails_and_strides(self):
        torch.manual_seed(419)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for lengths, dim, group in (((1, 9, 37), 10, 24), ((7, 8, 9, 25, 1), 11, 8),
                                       ((1, 129), 17, 128)):
                offsets = [0]
                for n in lengths:
                    offsets.append(offsets[-1] + n)
                # Strided channels and distinct head ranges expose wrong addressing/group boundaries.
                x = torch.randn(offsets[-1], dim*2, device=self.device, dtype=dtype)[:, ::2]
                for h, (start, end) in enumerate(zip(offsets, offsets[1:])):
                    x[start:end] *= h+1
                    x[start, :] = 2
                    x[start:end, -1] = .125
                for axis in ('channel', 'token'):
                    for bits in (16, 8, 4, 2):
                        with self.subTest(dtype=dtype, lengths=lengths, axis=axis, bits=bits):
                            reference = CompactPackedKV.quantize(x, offsets, group, axis=axis, bits=bits, backend='torch')
                            before = x.clone()
                            with patch.object(PackedKV, 'quantize', side_effect=AssertionError('per-head quantize')), \
                                 patch.object(torch, 'cat', side_effect=AssertionError('payload concatenation')), \
                                 patch.object(torch, 'isfinite', side_effect=AssertionError('separate finite scan')):
                                actual = CompactPackedKV.quantize(x, offsets, group, axis=axis, bits=bits)
                            self.assert_packed_equal(actual, reference)
                            torch.testing.assert_close(x, before, rtol=0, atol=0)
                            for h in range(len(lengths)):
                                a, b = actual.head(h), reference.head(h)
                                torch.testing.assert_close(a.dequantize(dtype=dtype, backend='torch'),
                                                           b.dequantize(dtype=dtype, backend='torch'), rtol=0, atol=0)

    def test_progressive_quantization_and_range_clamps(self):
        x = torch.tensor([[-1.e-7, 0., 1.e-7], [2., 2., 2.], [-1.e10, 0., 1.e10],
                          [1., -1., .5], [0., 0., 0.]], device=self.device)
        for axis in ('channel', 'token'):
            for group in (8, 4096):
                expected = CompactPackedKV.quantize(x, (0, 2, 5), group, axis=axis, bits=16, backend='torch')
                actual = CompactPackedKV.quantize(x, (0, 2, 5), group, axis=axis, bits=16)
                for bits in (16, 8, 4, 2):
                    self.assert_packed_equal(actual, expected)
                    direct = CompactPackedKV.quantize(x, (0, 2, 5), group, axis=axis, bits=bits)
                    self.assert_packed_equal(direct, expected)
                    before = actual.data.clone()
                    old = actual
                    actual = actual.shrink(backend='torch' if INTERPRETER else 'auto')
                    expected = expected.shrink(backend='torch')
                    torch.testing.assert_close(old.data, before, rtol=0, atol=0)

    def test_shared_arithmetic_preserves_rectangular_quantization(self):
        import triton
        from model.suprkv.packed_kernels import _quantize
        torch.manual_seed(619)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            value = torch.randn(2, 10, 9, device=self.device, dtype=dtype).transpose(1, 2)
            for axis in ('channel', 'token'):
                x = value.transpose(1, 2) if axis == 'channel' else value
                heads, rows, length = x.shape
                for group in (8, 24):
                    for bits in (16, 8, 4, 2):
                        ref = PackedKV.quantize(value, group, axis=axis, bits=bits, backend='torch')
                        data, scale, offset = (torch.empty_like(t) for t in (ref.data, ref.scale, ref.offset))
                        groups = triton.cdiv(length, group)
                        _quantize[(heads*rows, groups)](
                            x, data, scale, offset, length, triton.cdiv(length, 8)*8, rows, groups,
                            group, *x.stride(), bits, triton.next_power_of_2(group), enable_fp_fusion=False)
                        for actual, expected in ((data, ref.data), (scale, ref.scale), (offset, ref.offset)):
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_compact_rejects_nonfinite_values_and_propagates_oom(self):
        for axis in ('channel', 'token'):
            for bad in (float('nan'), float('inf'), -float('inf')):
                x = torch.ones(15, 9, device=self.device)
                x[-1, -1] = bad
                with self.assertRaisesRegex(ValueError, 'finite'):
                    CompactPackedKV.quantize(x, (0, 1, 6, 15), 8, axis=axis)
        with patch('model.suprkv.packed_kernels.quantize_compact', side_effect=torch.cuda.OutOfMemoryError('test OOM')):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                CompactPackedKV.quantize(torch.ones(3, 8, device=self.device), (0, 1, 3), 8, axis='token')

    def test_query_mean_fp32_accumulation_strides_and_no_full_cast(self):
        torch.manual_seed(47)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for heads, tokens, dim in ((3, 1, 1), (4, 7, 33), (28, 196, 128), (2, 257, 40), (1, 4096, 17)):
                with self.subTest(dtype=dtype, shape=(heads, tokens, dim)):
                    q = torch.randn(heads, tokens, dim*2, device=self.device, dtype=dtype)[..., ::2]
                    original = q.clone()
                    expected = q.float().mean(1)
                    fp64 = q.double().mean(1)
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16), \
                         patch.object(torch.Tensor, 'float', side_effect=AssertionError('full FP32 cast')):
                        actual = frame_query_mean(q)
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertFalse(actual.requires_grad)
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(actual.double(), fp64, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(q, original, rtol=0, atol=0)
        # FP16 accumulation would overflow; these are safe in FP32.
        q = torch.full((2, 196, 16), 60000., device=self.device, dtype=torch.float16)
        torch.testing.assert_close(frame_query_mean(q), q.float().mean(1), rtol=0, atol=0)
        # Broadcast strides (zero token/head stride) and constant input.
        q = torch.ones(1, 1, 5, device=self.device).expand(3, 196, 5)
        torch.testing.assert_close(frame_query_mean(q), torch.ones(3, 5, device=self.device), rtol=0, atol=0)

    def test_nonfinite_query_is_deferred_to_atomic_segment_fallback(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            key = torch.ones(2, 7, 8, device=self.device)
            cache = StreamingKVCache([(key[:, :2], key[:, :2])]*2, 100000, 7,
                                     dtype=torch.float32, compression_granularity='head')
            for frame in range(2):
                q = torch.ones(4, 7, 8, device=self.device)
                if frame == 1:
                    q[-1, -1, -1] = bad
                mean = frame_query_mean(q)
                torch.testing.assert_close(mean, q.float().mean(1), rtol=0, atol=0, equal_nan=True)
                cache.append_frame(frame, [(key, key)]*2, score_keys=[key, key],
                                   frame_queries=[torch.ones_like(mean), mean])
            originals = [f.layers for f in cache.frames.values()]
            cache.close_segment(None, scoring_strategy='frame_mean_query')
            self.assertEqual(cache.counts['scoring_failures'], 1)
            for frame, layers in zip(cache.frames.values(), originals):
                self.assertIs(frame.layers, layers)
            self.assertFalse(cache._score_keys)
            self.assertFalse(cache._score_queries)
            cache.audit_accounting()

    def test_encode_pre_rope_capture_commit_snapshot_and_budget(self):
        torch.manual_seed(87)
        from model.suprkv import scoring_kernels, packed_kernels
        rope = RotaryEmbeddingESM(16).to(self.device)
        sink = torch.zeros(2, 2, 16, device=self.device)
        def make_cache():
            return StreamingKVCache([(sink, sink)], 100000, 7, dtype=torch.float32,
                                    recent_frames=1, group_size=8, compression_granularity='head')
        actual, reference = make_cache(), make_cache()
        forward = streaming_attention_forward(64, 2, fattn=False)
        flatten = lambda x: x.transpose(1, 2).flatten(-2)
        with patch.object(scoring_kernels, 'frame_query_mean', wraps=scoring_kernels.frame_query_mean) as capture:
            for frame in range(3):
                q, k, v = (torch.randn(1, h, 7, 16, device=self.device) for h in (4, 2, 2))
                context = EncodeLayer(RotaryCache(sink[None], sink[None], torch.arange(2, device=self.device)),
                                      capture_frame_query=True)
                output, _ = forward(SimpleNamespace(), flatten(q), flatten(k), rope, True, context,
                                    lambda _: flatten(q), lambda _: flatten(k), lambda _: flatten(v), lambda x: x,
                                    16, 4, 2)
                torch.testing.assert_close(context.frame_query, q[0].float().mean(1), rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(context.score_key, k[0], rtol=0, atol=0)
                for cache, mean in ((actual, context.frame_query), (reference, q[0].float().mean(1))):
                    cache.append_frame(frame, [context.captured], score_keys=[context.score_key],
                                       frame_queries=[mean], rope_start=context.position_start)
                self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(capture.call_count, 3)
        snapshot = actual.acquire_snapshot(1)
        actual.close_segment(None, scoring_strategy='frame_mean_query')
        with patch('model.suprkv.segments._fused_frame_scores_supported', return_value=False), \
             patch('model.suprkv.cache._fused_partition_supported', return_value=False):
            reference.close_segment(None, scoring_strategy='frame_mean_query')
        self.assertTrue(all(not f.closed for f in snapshot.frames))
        actual.release_snapshot(1)
        for a, b in zip(actual.frames.values(), reference.frames.values()):
            for name in ('high_indices', 'low_indices', 'high_k', 'high_v', 'low_k', 'low_v'):
                torch.testing.assert_close(getattr(a.layers[0], name), getattr(b.layers[0], name), rtol=0, atol=0)
        self.assertEqual(actual.persistent_bytes(), reference.persistent_bytes())
        with patch.object(packed_kernels, 'quantize_compact', wraps=packed_kernels.quantize_compact) as quantize:
            actual.budget = actual.persistent_bytes() // 2
            actual._enforce_budget()
            self.assertGreater(quantize.call_count, 0)
        with patch('model.suprkv.quantization._compact_quantization_supported', return_value=False):
            reference.budget = reference.persistent_bytes() // 2
            reference._enforce_budget()
        self.assertEqual(actual.persistent_bytes(), reference.persistent_bytes())
        self.assertEqual(dict(actual.counts), dict(reference.counts))
        self.assertEqual(tuple(actual.frames), tuple(reference.frames))
        self.assertEqual(actual.last_event['quantized'], reference.last_event['quantized'])
        for a, b in zip(actual.frames.values(), reference.frames.values()):
            self.assertEqual(a.layers[0].bits, b.layers[0].bits)
            if a.layers[0].bits < 16:
                self.assert_packed_equal(a.layers[0].high_k, b.layers[0].high_k)
                self.assert_packed_equal(a.layers[0].high_v, b.layers[0].high_v)
        actual.audit_accounting()
        reference.audit_accounting()


if __name__ == '__main__':
    unittest.main()
