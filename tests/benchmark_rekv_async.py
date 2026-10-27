"""Compare real single-GPU ReKV stream runs; NVTX ranges support Nsight inspection.

Run from the repository root: python -m tests.benchmark_rekv_async --help
No scoring service is called. Models and videos must already be available locally.
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import time

import torch

from model.rekv.async_runtime import ReKVAsyncRuntime, single_cuda_device
from video_qa.methods import get_method
from video_qa.methods.rekv import async_pending_groups
from video_qa.model_paths import MODEL_PATHS
from video_qa.rekv_stream_vqa import ReKVStreamVQA
from video_qa.sampling import load_samples


class MeasuredRuntime(ReKVAsyncRuntime):
    def __init__(self, model, submitted):
        super().__init__(model)
        self.submitted = submitted

    def capture(self):
        start = time.perf_counter()
        snapshot = super().capture()
        self.submitted[id(snapshot)] = start
        return snapshot

    def answer(self, callback, video_sample, samples, snapshot):
        try:
            return super().answer(callback, video_sample, samples, snapshot)
        finally:
            self.submitted.pop(id(snapshot), None)


class MeasuredVQA(ReKVStreamVQA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qa_times = []
        self.submitted = {}
        self.runtime_factory = lambda model: MeasuredRuntime(model, self.submitted)

    def consume_frame(self, frame, timestamp):
        with torch.cuda.nvtx.range('rekv.encode_frame'):
            return super().consume_frame(frame, timestamp)

    def answer_record(self, video_sample, sample, **kwargs):
        start = time.perf_counter()
        # All questions in a group become eligible at the same prefix publication.
        submitted = self.submitted.get(id(kwargs.get('kv_snapshot')), start)
        with torch.cuda.nvtx.range('rekv.answer'):
            result = super().answer_record(video_sample, sample, **kwargs)
            torch.cuda.current_stream(self.qa_model.device).synchronize()
        end = time.perf_counter()
        self.qa_times.append(dict(video_id=video_sample['video_id'], question=sample['question'],
                                  execution_seconds=end - start, snapshot_and_queue_seconds=start - submitted,
                                  response_seconds=end - submitted))
        return result


def run(model, samples, args, asynchronous):
    random.seed(2024)
    torch.manual_seed(2024)
    # Start each trial with the same allocator state, outside the timer. Video
    # cleanup in both modes releases references without flushing the allocator.
    model.clear_cache(release_allocator=True)
    torch.cuda.synchronize(model.device)
    torch.cuda.reset_peak_memory_stats(model.device)
    analyzer = MeasuredVQA(anno=samples, qa_model=model, save_dir='.', sample_fps=args.sample_fps,
                           retrieve_size=args.retrieve_size or 64, chunk_size=args.retrieve_chunk_size or 1,
                           rekv_async=asynchronous,
                           rekv_async_max_pending_groups=async_pending_groups(args))
    start = time.perf_counter()
    with torch.cuda.nvtx.range('rekv.async' if asynchronous else 'rekv.sync'):
        for sample in samples:
            analyzer.analyze_a_video(sample)
        torch.cuda.synchronize(model.device)
    result = dict(mode='async' if asynchronous else 'sync', seconds=time.perf_counter() - start,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(model.device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(model.device),
                  qa_execution_times=analyzer.qa_times, records=analyzer.records,
                  async_video_stats=analyzer.async_video_stats)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=[name for name in MODEL_PATHS if name != 'llava_ov_72b'],
                        default='llava_ov_7b')
    parser.add_argument('--anno_path', required=True)
    parser.add_argument('--sample_fps', type=float, default=.5)
    parser.add_argument('--max_samples', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    method = get_method('rekv')
    method.add_arguments(parser)
    args = parser.parse_args()
    # This benchmark always runs an async leg as well as the synchronous reference.
    args.rekv_async = True
    if args.repeat < 1 or args.warmup < 0 or args.max_samples < 0 or args.sample_fps <= 0:
        parser.error('repeat/sample_fps must be positive; warmup/max_samples must be nonnegative')
    if not torch.cuda.is_available():
        parser.error('A working CUDA GPU is required for this benchmark')
    method.validate(args)
    samples = load_samples(args.anno_path, args.max_samples)
    if not samples or not any(s['conversations'] for s in samples):
        parser.error('Annotations must contain at least one video and question')
    model, _ = method.load(args)
    single_cuda_device(model)
    measurements = []
    reference = None
    try:
        for _ in range(args.warmup):
            for asynchronous in (False, True):
                run(model, samples, args, asynchronous)
        for repetition in range(args.repeat):
            # Alternate order to reduce warm cache/thermal ordering bias.
            for asynchronous in ((False, True) if repetition % 2 == 0 else (True, False)):
                result = run(model, samples, args, asynchronous)
                if reference is None:
                    reference = result['records']
                result['matches_reference'] = result['records'] == reference
                result['repetition'] = repetition
                measurements.append(result)
    finally:
        model.clear_cache(release_allocator=True)
    summary = {}
    for mode in ('sync', 'async'):
        runs = [r for r in measurements if r['mode'] == mode]
        summary[mode] = dict(median_seconds=statistics.median(r['seconds'] for r in runs),
                             peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in runs))
    report = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  gpu=torch.cuda.get_device_name(model.device), torch_version=torch.__version__,
                  qa_latency_definition='response: prefix capture start to GPU completion; execution: QA start to GPU completion',
                  producer_time_definition='host time to submit all frames/question groups; includes backpressure, excludes final answer drain',
                  total_time_definition='video initialization, encoding, QA, and video-state cleanup in both modes; allocator trimming excluded',
                  summary=summary, runs=measurements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'Report: {args.output}')
    if not all(r['matches_reference'] for r in measurements):
        raise SystemExit('Generated records differ; inspect the report before accepting the async run')


if __name__ == '__main__':
    main()
