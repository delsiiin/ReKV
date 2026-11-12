"""Fixed-input, synchronized per-head versus batched encoding attention."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model.suprkv.encoding_attention import attend
from test_suprkv_encoding_attention import inputs, legacy


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    torch.manual_seed(351)
    torch.set_num_threads(1)
    report = dict(gpu=torch.cuda.get_device_name(), dtype='bfloat16', frame_tokens=196,
                  query_heads=28, kv_heads=4, dim=128, repeats=args.repeats, cases=[])
    for lengths in ((4096, 4096, 4096, 4096), (2048, 4096, 8192, 16384)):
        q, k, v, heads = inputs(lengths, 196, 128, 7, torch.bfloat16)
        # Production prefixes have contiguous token rows.
        for h in heads:
            h.key, h.value = h.key.contiguous(), h.value.contiguous()
        calls = dict(reference=lambda: legacy(q, k, v, heads), batched=lambda: attend(q, k, v, heads))
        for call in calls.values():
            for _ in range(3):
                call()
        torch.testing.assert_close(calls['reference'](), calls['batched'](), rtol=0, atol=0)
        times = {name: [] for name in calls}
        for repeat in range(args.repeats):
            order = ('reference', 'batched') if repeat % 2 == 0 else ('batched', 'reference')
            for name in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                calls[name]()
                torch.cuda.synchronize()
                times[name].append((time.perf_counter()-start)*1000)
        medians = {name: statistics.median(values) for name, values in times.items()}
        report['cases'].append(dict(prefix_lengths=lengths, milliseconds=times, median_ms=medians,
                                    exact_match=True, speedup=medians['reference']/medians['batched']))
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
