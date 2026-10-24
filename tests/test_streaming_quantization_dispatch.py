"""Regression tests for ragged-cache JIT explosion and fused initialization."""
import unittest

import torch

from model.streaming.quantization import PackedKV


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class QuantizationDispatchTests(unittest.TestCase):
    @torch.inference_mode()
    def test_ragged_shapes_reuse_compiled_kernels(self):
        from model.streaming import packed_kernels as kernels
        functions = (kernels._quantize, kernels._shrink, kernels._decode)
        def variants():
            return tuple(len(f.device_caches[torch.cuda.current_device()][0]) for f in functions)
        def exercise(heads, tokens):
            x = torch.randn(heads, tokens, 64, device='cuda')
            for axis in ('channel', 'token'):
                for initial_bits in (16, 8):
                    actual = PackedKV.quantize(x, 24, axis=axis, bits=initial_bits, backend='triton')
                    reference = PackedKV.quantize(x, 24, axis=axis, bits=initial_bits, backend='torch')
                    while True:
                        torch.testing.assert_close(actual.data, reference.data, rtol=0, atol=0)
                        torch.testing.assert_close(actual.scale, reference.scale, rtol=0, atol=0)
                        torch.testing.assert_close(actual.dequantize(dtype=torch.float32, backend='triton'),
                                                   reference.dequantize(dtype=torch.float32, backend='torch'), rtol=0, atol=0)
                        if actual.bits == 2:
                            break
                        actual, reference = actual.shrink(backend='triton'), reference.shrink(backend='torch')
        exercise(2, 1)
        compiled = variants()
        # Alternate growth and shrinkage; include word/group/tile boundaries
        # and sequences substantially longer than a single video frame.
        shapes = ((1, 7), (3, 1024), (3, 8), (2, 257), (2, 9), (1, 23),
                  (2, 4097), (2, 24), (3, 25), (1, 1), (2, 128), (1, 129), (2, 196))
        for heads, tokens in shapes:
            with self.subTest(heads=heads, tokens=tokens):
                exercise(heads, tokens)
                self.assertEqual(variants(), compiled, 'A new retained length must not compile new kernels')

    @torch.inference_mode()
    def test_fused_initialization_matches_sequential_reference(self):
        torch.manual_seed(89)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for axis in ('channel', 'token'):
                for group in (8, 24, 128):
                    x = torch.randn(2, 57, 11, dtype=dtype, device='cuda').transpose(1, 2)
                    x[:, 0] = 2
                    reference = PackedKV.quantize(x, group, axis=axis, backend='torch')
                    for bits in (8, 4, 2):
                        reference = reference.shrink(backend='torch')
                        actual = PackedKV.quantize(x, group, axis=axis, bits=bits, backend='triton')
                        torch.testing.assert_close(actual.data, reference.data, rtol=0, atol=0)
                        torch.testing.assert_close(actual.scale, reference.scale, rtol=0, atol=0)
                        torch.testing.assert_close(actual.offset, reference.offset, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
