"""Synthetic CUDA latency/memory comparison; excludes model weights and JIT.

python tests/benchmark_streaming_packed.py --device cuda:1 --tokens 4096
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from model.attention.rope import RotaryEmbeddingESM
from model.attention.dot_production_attention import get_multi_stage_dot_production_attention
from model.streaming.packed_attention import AttentionChunk, PackedRotaryCache, coalesce
from model.streaming.quantization import PackedKV
from model.streaming.rotary import rotate_at


def measure(fn, repeats):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1000)
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()-before
    del result
    return dict(ms=round(statistics.median(times), 4), peak_mib=round(peak/2**20, 3))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--tokens', type=int, default=4096)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--queries', type=int, nargs='+', default=[1, 16])
    parser.add_argument('--bits', type=int, nargs='+', default=[8, 4, 2])
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if args.tokens < args.frames or args.tokens % args.frames:
        parser.error('--tokens must be a positive multiple of --frames')
    torch.cuda.set_device(args.device)
    torch.manual_seed(71)
    dtype, h, hkv, dim = torch.float16, 16, 4, 128
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, **vars(args))))
    rope = RotaryEmbeddingESM(dim).to(args.device)
    Attn, _ = get_multi_stage_dot_production_attention(True)
    for bits in args.bits:
        k = PackedKV.quantize(torch.randn(hkv, args.tokens, dim, device=args.device, dtype=dtype),
                              128, axis='channel', bits=bits)
        v = PackedKV.quantize(torch.randn(hkv, args.tokens, dim, device=args.device, dtype=dtype),
                              128, bits=bits)
        for m in args.queries:
            q = torch.randn(1, h, m, dim, device=args.device, dtype=dtype)
            p = torch.randn(1, h, m, args.tokens, device=args.device, dtype=dtype)
            for name, packed, a, transpose in [('QK', k, q, True), ('PV', v, p, False)]:
                for backend in ('torch', 'triton'):
                    result = measure(lambda: packed.matmul(a, transpose=transpose, backend=backend), args.repeats)
                    print(json.dumps(dict(op=name, bits=bits, queries=m, backend=backend, **result)))

        # Realistic per-frame packing and position-delta rotation. Compare one
        # attention call including materialization, then repeated resident decode.
        chunks = []
        size = args.tokens // args.frames
        for i in range(args.frames):
            key = PackedKV.quantize(torch.randn(hkv, size, dim, device=args.device, dtype=dtype),
                                    128, axis='channel', bits=bits)
            value = PackedKV.quantize(torch.randn(hkv, size, dim, device=args.device, dtype=dtype), 128, bits=bits)
            ranks = torch.arange(i*size, (i+1)*size, device=args.device)
            chunks.append(AttentionChunk.create(key, value, ranks, ranks+19, rope))
        cache = PackedRotaryCache(coalesce(chunks), args.tokens, dtype)
        delta = torch.full((args.tokens,), -19, device=args.device, dtype=torch.long)
        def prepare_dense():
            # Decode each frame, then rotate the joined K once, as in the
            # production dense path. Do not charge per-frame RoPE launches.
            key = torch.cat([c.key.dequantize(dtype=dtype) for c in chunks], 1)[None]
            value = torch.cat([c.value.dequantize(dtype=dtype) for c in chunks], 1)[None]
            return rotate_at(key, delta, rope), value
        for m in args.queries:
            q = torch.randn(1, h, m, dim, device=args.device, dtype=dtype)
            nk = torch.randn(1, hkv, m, dim, device=args.device, dtype=dtype)
            nv = torch.randn_like(nk)
            pos = torch.arange(args.tokens, args.tokens+m, device=args.device)
            nk, qr = rotate_at(nk, pos, rope), rotate_at(q, pos, rope)
            dense_k, dense_v = prepare_dense()
            def dense_attention(key, value):
                attn = Attn(q.shape, q.dtype, q.device)
                attn.append(qr, torch.cat((key, nk), -2), torch.cat((value, nv), -2),
                            sliding_window=args.tokens+m, end=True)
                return attn.get_result()[0]
            def materialized():
                key, value = prepare_dense()
                return dense_attention(key, value)
            fused = lambda: cache.append(nk, nv, rope).attend(q, rope)
            def prepared():
                return PackedRotaryCache(coalesce(chunks), args.tokens, dtype).append(nk, nv, rope).attend(q, rope)
            torch.testing.assert_close(fused(), dense_attention(dense_k, dense_v), rtol=.008, atol=.008)
            for name, fn in [('materialize_attention', materialized),
                             ('resident_dense_attention', lambda: dense_attention(dense_k, dense_v)),
                             ('prepare_packed_attention', prepared),
                             ('packed_attention', fused)]:
                result = measure(fn, args.repeats)
                print(json.dumps(dict(op=name, bits=bits, queries=m, **result)))


if __name__ == '__main__':
    main()
