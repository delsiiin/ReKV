"""Replay ragged compression shapes, including first-use compilation costs.

Use --observation-result to replay retained per-layer lengths from a saved run.
--cache-dir should be a fresh directory when measuring cold compilation.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--observation-result', type=Path)
    parser.add_argument('--cache-dir')
    parser.add_argument('--baseline-dir', type=Path, help='Directory containing packed_kernels_before.py and quantization_before.py')
    parser.add_argument('--backend', choices=['auto', 'torch'], default='auto')
    args = parser.parse_args()
    if args.cache_dir:
        os.environ['TRITON_CACHE_DIR'] = args.cache_dir
    import torch
    from model.streaming import packed_kernels
    from model.streaming.quantization import PackedKV
    if args.baseline_dir:
        packed_kernels = load_module('model.streaming.packed_kernels', args.baseline_dir/'packed_kernels_before.py')
        PackedKV = load_module('model.streaming._baseline_quantization', args.baseline_dir/'quantization_before.py').PackedKV
    if args.observation_result:
        result = json.loads(args.observation_result.read_text())['compression_experiment']
        states = result['runs']['after']['timeline'][-1]['frame_states']['frames']
        records = [(sum(c[1:]), next(b for b, n in zip((8, 4, 2), c[1:]) if n))
                   for f in states for c in f.get('counts') or [] if sum(c[1:])]
        group = result['config']['group_size']
    else:
        records = [(n, 4) for n in range(70, 115)] * 8
        group = 128
    torch.cuda.set_device(args.device)
    torch.manual_seed(83)
    dtype = torch.bfloat16
    inputs = {n: torch.randn(4, n, 128, device=args.device, dtype=dtype) for n, _ in records}
    torch.cuda.synchronize()
    # Initialize Triton's driver before timing: report per-kernel specialization
    # counts, independent of whether disk cache happens to exist.
    kernels = [packed_kernels._quantize, packed_kernels._shrink, packed_kernels._decode]
    def variants():
        return {fn.__name__: len(fn.device_caches[torch.cuda.current_device()][0]) for fn in kernels}
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), records=len(records),
                          distinct_lengths=len(inputs), group=group, backend=args.backend,
                          baseline=bool(args.baseline_dir))), flush=True)
    def timed(label, fn):
        before = variants()
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        after = variants()
        print(json.dumps(dict(stage=label, seconds=elapsed, variants=after,
                              new_variants={k: after[k]-before[k] for k in before})), flush=True)
        return value
    with torch.inference_mode():
        for pass_name in ('first', 'repeat'):
            def quantize():
                return [tuple(PackedKV.quantize(inputs[n], group, axis=axis, bits=8, backend=args.backend)
                              for axis in ('channel', 'token')) for n, _ in records]
            payloads = timed(pass_name+'_initial_int8', quantize)
            def shrink():
                for i, (_, bits) in enumerate(records):
                    while payloads[i][0].bits > bits:
                        payloads[i] = tuple(p.shrink(backend=args.backend) for p in payloads[i])
            timed(pass_name+'_progressive', shrink)
            def decode():
                for pair in payloads:
                    for p in pair:
                        p.dequantize(dtype=dtype, backend=args.backend)
            timed(pass_name+'_decode', decode)
            del payloads


if __name__ == '__main__':
    main()
