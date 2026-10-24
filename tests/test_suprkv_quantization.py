"""Packed kernels against the PM-KVQ-compatible PyTorch reference."""
import importlib.util
import unittest
from unittest.mock import patch

import torch

from model.suprkv.quantization import PackedKV, pack


class PackedMatmulTests(unittest.TestCase):
    def test_cpu_products_and_validation(self):
        torch.manual_seed(21)
        x = torch.randn(2, 11, 10)
        p = PackedKV.quantize(x, 8, bits=8)
        for bits in (8, 4, 2):
            self.assertEqual(p.bits, bits)
            for transpose in (False, True):
                left = torch.randn(2, 4, 3, 10 if transpose else 11)
                dense = p.dequantize(dtype=left.dtype).repeat_interleave(2, 0)
                expected = left @ (dense.transpose(-1, -2) if transpose else dense)
                torch.testing.assert_close(p.matmul(left, transpose=transpose), expected)
            p = p.shrink()
        for left in (torch.ones(3, 1, 11), torch.ones(4, 1, 12), torch.ones(4, 1, 11, dtype=torch.int32)):
            with self.assertRaises(ValueError):
                p.matmul(left)
        with self.assertRaises(ValueError):
            p.dequantize(backend='invalid')
        with self.assertRaises(ValueError):
            p.dequantize(backend='triton')


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec('triton'), 'CUDA/Triton required')
class PackedCudaTests(unittest.TestCase):
    def test_exhaustive_unsigned_int16_shrinking(self):
        codes = torch.arange(65536, device='cuda', dtype=torch.int32).view(1, 1, -1)
        scale = torch.ones(1, 1, 512, device='cuda')
        ref = PackedKV(pack(codes, 16), scale, torch.zeros_like(scale), (1, 1, 65536), 16, 128)
        actual = ref
        for bits in (8, 4, 2):
            ref, actual = ref.shrink(backend='torch'), actual.shrink(backend='triton')
            self.assertEqual(actual.bits, bits)
            torch.testing.assert_close(actual.data, ref.data, rtol=0, atol=0)
            torch.testing.assert_close(actual.scale, ref.scale, rtol=0, atol=0)

    def test_progressive_kernels_exact_codes_parameters_and_decode(self):
        torch.manual_seed(31)
        # Non-power-of-two groups, padding, constant groups, and small ranges.
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for channels, group in ((128, 128), (10, 8), (57, 24), (10, 128)):
                x = torch.randn(2, channels, 9, device='cuda', dtype=dtype).transpose(1, 2)
                x[:, 0] = 2
                x[:, 1] *= 1.e-4
                ref = PackedKV.quantize(x, group, backend='torch')
                actual = PackedKV.quantize(x, group, backend='triton')
                fused = PackedKV.quantize(x, group, bits=8, backend='triton')
                for bits in (16, 8, 4, 2):
                    with self.subTest(dtype=dtype, channels=channels, group=group, bits=bits):
                        torch.testing.assert_close(actual.data, ref.data, rtol=0, atol=0)
                        torch.testing.assert_close(actual.scale, ref.scale, rtol=0, atol=0)
                        torch.testing.assert_close(actual.offset, ref.offset, rtol=0, atol=0)
                        for out_dtype in (torch.float16, torch.bfloat16, torch.float32):
                            torch.testing.assert_close(actual.dequantize(dtype=out_dtype, backend='triton'),
                                                       ref.dequantize(dtype=out_dtype, backend='torch'), rtol=0, atol=0)
                        if bits == 8:
                            torch.testing.assert_close(fused.data, ref.data, rtol=0, atol=0)
                            torch.testing.assert_close(fused.scale, ref.scale, rtol=0, atol=0)
                        old_data, old_scale = actual.data.clone(), actual.scale.clone()
                        previous = actual
                        actual = actual.shrink(backend='triton')
                        torch.testing.assert_close(previous.data, old_data, rtol=0, atol=0)
                        torch.testing.assert_close(previous.scale, old_scale, rtol=0, atol=0)
                        ref = ref.shrink(backend='torch')

    def test_fused_products_gqa_tails_and_no_dense_decode(self):
        torch.manual_seed(41)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for channels, group in ((128, 128), (10, 8), (57, 24)):
                p = PackedKV.quantize(torch.randn(2, 37, channels, device='cuda'), group)
                for bits in (16, 8, 4, 2):
                    for transpose in (False, True):
                        reduction = channels if transpose else 37
                        # Noncontiguous input, multi-batch, GQA, and partial tiles.
                        left = torch.randn(2, 4, reduction, 19, device='cuda', dtype=dtype).transpose(-1, -2)
                        expected = p.matmul(left, transpose=transpose, backend='torch')
                        with patch.object(PackedKV, 'dequantize', side_effect=AssertionError('No dense KV allocation')):
                            actual = p.matmul(left, transpose=transpose, backend='triton')
                        tol = 2.e-2 if dtype == torch.bfloat16 else 2.e-3 if dtype == torch.float16 else 3.e-5
                        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
                    p = p.shrink()

    @unittest.skipUnless(torch.cuda.device_count() > 1, 'Multiple GPUs required')
    def test_noncurrent_cuda_device(self):
        with torch.cuda.device(0):
            x = torch.randn(2, 7, 10, device='cuda:1')
            p = PackedKV.quantize(x, 8, bits=8).shrink()
            self.assertEqual(p.dequantize().device, x.device)
            q = torch.randn(4, 1, 10, device=x.device, dtype=torch.float16)
            torch.testing.assert_close(p.matmul(q, transpose=True),
                                       p.matmul(q, transpose=True, backend='torch'), rtol=.002, atol=.002)


if __name__ == '__main__':
    unittest.main()
