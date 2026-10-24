"""Metadata benchmark; optional replay includes token scoring and updates.

No model load, summary generation or quantization is timed.

python tests/benchmark_suprkv_accounting.py --baseline /tmp/cache-before.py
The optional baseline is a saved, pre-change model/suprkv/cache.py.
"""
import argparse
from dataclasses import replace
import importlib.util
import inspect
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from model.suprkv.cache import StreamingKVCache


def load_baseline(path):
    name = 'model.suprkv._accounting_benchmark_baseline'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.StreamingKVCache


def median_ms(action, repeats, prepare=lambda: None):
    values = []
    for i in range(repeats+2):
        prepare()
        start = time.perf_counter()
        action()
        duration = (time.perf_counter()-start)*1000
        if i >= 2:
            values.append(duration)
    return statistics.median(values)


def run(cls, frames, layers, repeats, device):
    x = torch.zeros(4, 8, 128, dtype=torch.float16, device=device)
    cache = cls([(x[:, :2], x[:, :2])]*layers, 1000000, 8)
    start = time.perf_counter()
    for i in range(frames):
        cache.append_frame(i, [(x, x)]*layers)
    append_ms = (time.perf_counter()-start)*1000/frames
    # Distinct high/low tensors reproduce the production object count cheaply.
    for i, frame in list(cache.frames.items()):
        classified = tuple(replace(layer,
            high_indices=layer.high_indices[:4].clone(), low_indices=layer.high_indices[4:].clone(),
            high_k=layer.high_k[:, :4].clone(), high_v=layer.high_v[:, :4].clone(),
            low_k=layer.high_k[:, 4:].clone(), low_v=layer.high_v[:, 4:].clone()) for layer in frame.layers)
        cache.frames[i] = replace(frame, layers=classified, closed=True)
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    def prepare_commit():
        cache.frames[frames-1] = replace(cache.frames[frames-1], closed=False)
    def commit_and_stats():
        # Include a second read, matching the previous engine call pattern.
        cache.close_segment(None)
        cache.memory_stats()
    result = dict(frames=frames, layers=layers, append_ms_per_frame=append_ms,
        persistent_ms=median_ms(cache.persistent_bytes, repeats),
        stats_ms=median_ms(cache.memory_stats, repeats),
        frame_states_ms=median_ms(cache.frame_states, repeats),
        commit_ms=median_ms(commit_and_stats, repeats, prepare_commit))
    if hasattr(cache, 'audit_accounting'):
        cache.audit_accounting()
    return result


def replay(cls, frames, layers, device):
    x = torch.arange(8, device=device).view(1, 8, 1).expand(4, 8, 128).half()
    cache = cls([(x[:, :2], x[:, :2])]*layers, 1000000, 8)
    queries = [torch.ones(2, 8, 128, device=device)]*layers
    score_kwargs = {'score_keys': [x]*layers} if 'score_keys' in inspect.signature(cache.append_frame).parameters else {}
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    commits = 0
    for i in range(frames):
        cache.append_frame(i, [(x, x)]*layers, **score_kwargs)
        if i % 2 == 1 or i == frames-1:
            cache.close_segment(queries)
            cache.memory_stats()
            commits += 1
        cache.memory_stats()
        cache.frame_states()
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    seconds = time.perf_counter()-start
    if hasattr(cache, 'audit_accounting'):
        cache.audit_accounting()
    return dict(frames=frames, layers=layers, commits=commits, replay_seconds=seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--frames', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('--layers', type=int, default=28)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--replay', action='store_true', help='Also time append/scoring/commit/reporting together')
    args = parser.parse_args()
    torch.set_num_threads(1)
    classes = {'incremental': StreamingKVCache}
    if args.baseline:
        classes = {'baseline': load_baseline(args.baseline), **classes}
    for name, cls in classes.items():
        for frames in args.frames:
            print(json.dumps(dict(implementation=name, device=args.device,
                                  **run(cls, frames, args.layers, args.repeats, args.device))), flush=True)
        if args.replay:
            print(json.dumps(dict(implementation=name, device=args.device,
                                  **replay(cls, max(args.frames), args.layers, args.device))), flush=True)


if __name__ == '__main__':
    main()
