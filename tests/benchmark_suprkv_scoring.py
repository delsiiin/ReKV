"""Synchronized segment-commit microbenchmark; no model, summary or quantization.

Compare head/frame_mean_query fusion with the torch reference:
python tests/benchmark_suprkv_scoring.py --compare-reference

An external cache-class baseline can also be supplied with --baseline.
It shares the currently installed scoring functions; --compare-reference
disables BOTH P0 optimizations and is the appropriate fusion baseline.
"""
import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model.suprkv.cache import StreamingKVCache


def load_baseline(path):
    name = 'model.suprkv._scoring_benchmark_baseline'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.StreamingKVCache


@torch.inference_mode()
def benchmark(classes, frames, layers, repeats, device, *, compression_granularity='head',
              scoring_strategy='frame_mean_query'):
    torch.manual_seed(24)
    keys = torch.randn(frames, layers, 4, 196, 128, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    queries = list(torch.randn(layers, frames if scoring_strategy == 'frame_mean_query' else 32,
                               28, 128, device=device,
                               dtype=torch.float32 if scoring_strategy == 'frame_mean_query' else torch.bfloat16))
    sink = keys[0, :, :, :8]
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    times = {name: [] for name in classes}
    peaks = {name: [] for name in classes}
    for trial in range(repeats + 2):
        # Alternate order to reduce systematic warmup/clock bias.
        for name in list(classes)[::1 if trial % 2 == 0 else -1]:
            cache = classes[name]([(x, x) for x in sink], 1000000, 196, dtype=torch.bfloat16,
                                  compression_granularity=compression_granularity)
            for frame in range(frames):
                cache.append_frame(frame, list(zip(keys[frame], values[frame])), score_keys=list(keys[frame]),
                                   frame_queries=[q[frame] for q in queries]
                                   if scoring_strategy == 'frame_mean_query' else None)
            with ExitStack() as dispatch:
                # Install reference dispatch before timing; mock/context setup
                # must not count toward the measured baseline latency.
                if name == 'torch_reference':
                    dispatch.enter_context(patch('model.suprkv.segments._fused_frame_scores_supported',
                                                 new=lambda *_: False))
                    dispatch.enter_context(patch('model.suprkv.cache._fused_partition_supported',
                                                 new=lambda *_: False))
                synchronize()
                if device.type == 'cuda':
                    before = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                cache.close_segment(queries if scoring_strategy == 'summary' else None, scoring_strategy=scoring_strategy)
                synchronize()
                elapsed = (time.perf_counter() - start) * 1000
            if trial >= 2:
                times[name].append(elapsed)
                if device.type == 'cuda':
                    peaks[name].append(torch.cuda.max_memory_allocated(device) - before)
            assert cache.counts['scoring_failures'] == 0
            assert cache.counts['compressions'] == 0
            del cache
    return dict(frames=frames, median_commit_ms={name: statistics.median(t) for name, t in times.items()},
                peak_extra_bytes={name: max(p) for name, p in peaks.items() if p})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline')
    parser.add_argument('--compare-reference', action='store_true')
    parser.add_argument('--compression-granularity', choices=('layer', 'head'), default='head')
    parser.add_argument('--scoring-strategy', choices=('summary', 'frame_mean_query'), default='frame_mean_query')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--frames', nargs='+', type=int, default=[1, 2, 4, 8, 32])
    parser.add_argument('--layers', type=int, default=28)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if min(*args.frames, args.layers, args.repeats) < 1:
        parser.error('Frame counts, layers and repeats must be positive')
    device = torch.device(args.device)
    torch.set_num_threads(1)
    classes = {'current': StreamingKVCache}
    if args.compare_reference:
        classes['torch_reference'] = StreamingKVCache
    if args.baseline:
        classes['before'] = load_baseline(args.baseline)
    print(json.dumps(dict(device=str(device), layers=args.layers, repeats=args.repeats,
                          compression_granularity=args.compression_granularity, scoring_strategy=args.scoring_strategy,
                          gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None)), flush=True)
    for frames in args.frames:
        print(json.dumps(benchmark(classes, frames, args.layers, args.repeats, device,
                                   compression_granularity=args.compression_granularity,
                                   scoring_strategy=args.scoring_strategy)), flush=True)


if __name__ == '__main__':
    main()
