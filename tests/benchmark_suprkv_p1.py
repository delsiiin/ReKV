"""Warm P1 microbenchmarks: fused frame means and compact initial INT8.

Reference compact quantization uses the previous per-head dispatch (including
Triton on CUDA), so the comparison isolates batching instead of changing both
batching and the per-head quantizer. No model or JIT time is included.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model.suprkv.quantization import CompactPackedKV, _compact_quantization_supported
from model.suprkv.segments import frame_query_mean, _fused_frame_query_mean_supported


@torch.inference_mode()
def benchmark(args):
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    torch.manual_seed(91)
    offsets = [0]
    for n in args.lengths:
        offsets.append(offsets[-1]+n)
    value = torch.randn(offsets[-1], args.dim, dtype=dtype, device=device)
    query = torch.randn(len(args.lengths)*args.gqa, args.frame_tokens, args.dim, dtype=dtype, device=device)
    stages = {
        'frame_query_mean': (lambda: frame_query_mean(query),
                             'model.suprkv.segments._fused_frame_query_mean_supported'),
        'compact_key_int8': (lambda: CompactPackedKV.quantize(value, offsets, args.group_size, axis='channel'),
                            'model.suprkv.quantization._compact_quantization_supported'),
        'compact_value_int8': (lambda: CompactPackedKV.quantize(value, offsets, args.group_size, axis='token'),
                              'model.suprkv.quantization._compact_quantization_supported'),
    }
    print(json.dumps(dict(device=str(device), dtype=args.dtype, lengths=args.lengths, dim=args.dim,
                          frame_tokens=args.frame_tokens, group_size=args.group_size,
                          fused_mean_supported=_fused_frame_query_mean_supported(query),
                          fused_quantization_supported=_compact_quantization_supported(value, args.group_size),
                          gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None)), flush=True)
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    for stage, (fn, selector) in stages.items():
        times, peaks = {k: [] for k in ('fused', 'reference')}, {k: [] for k in ('fused', 'reference')}
        for trial in range(args.repeats+2):
            for name in list(times)[::1 if trial % 2 == 0 else -1]:
                # Dispatch setup is outside the timed interval.
                with patch(selector, new=lambda *_: False) if name == 'reference' else nullcontext():
                    sync()
                    if device.type == 'cuda':
                        before = torch.cuda.memory_allocated(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter()
                    result = fn()
                    sync()
                    elapsed = (time.perf_counter()-start)*1000
                    if trial >= 2:
                        times[name].append(elapsed)
                        if device.type == 'cuda':
                            peaks[name].append(torch.cuda.max_memory_allocated(device)-before)
                    del result
        print(json.dumps(dict(stage=stage, median_ms={k: statistics.median(v) for k, v in times.items()},
                              peak_extra_bytes={k: max(v) for k, v in peaks.items() if v})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', choices=('float16', 'bfloat16', 'float32'), default='bfloat16')
    parser.add_argument('--lengths', type=int, nargs='+', default=[70, 91, 103, 114])
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--group-size', type=int, default=128)
    parser.add_argument('--gqa', type=int, default=7)
    parser.add_argument('--frame-tokens', type=int, default=196)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if min(*args.lengths, args.dim, args.group_size, args.gqa, args.frame_tokens, args.repeats) < 1:
        parser.error('Shapes, group size and repeats must be positive')
    if args.group_size % 8:
        parser.error('Group size must be a multiple of eight')
    torch.set_num_threads(1)
    benchmark(args)


if __name__ == '__main__':
    main()
