"""ReKV stream QA entry point."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby, islice
import math
import time

from model.rekv.async_runtime import ReKVAsyncRuntime
from video_qa.base import work
from video_qa.flows import StreamVQA
from video_qa.methods.lifecycle import ReKVLifecycle
from video_qa.sampling import sampled_frames


class ReKVStreamVQA(ReKVLifecycle, StreamVQA):
    supports_rekv_async = True
    runtime_factory = ReKVAsyncRuntime

    def __init__(self, *args, rekv_async=False, rekv_async_max_pending_groups=2, **kwargs):
        if not isinstance(rekv_async_max_pending_groups, int) or rekv_async_max_pending_groups < 0:
            raise ValueError('rekv_async_max_pending_groups must be a nonnegative integer')
        self.rekv_async = rekv_async
        self.max_pending_groups = rekv_async_max_pending_groups
        self.async_video_stats = []
        super().__init__(*args, **kwargs)

    def frame_source(self, video_sample):
        return sampled_frames(video_sample['video_path'], self.sample_fps)

    def answer_group(self, video_sample, samples, **generation_options):
        """One reader owns a published prefix throughout this cutoff's questions."""
        return [self.answer_record(video_sample, sample, **generation_options) for sample in samples]

    def analyze_a_video(self, video_sample):
        if not self.rekv_async:
            try:
                return super().analyze_a_video(video_sample)
            finally:
                # Both modes release video state before returning to callers.
                self.qa_model.clear_cache()
        questions = video_sample['conversations']
        cutoffs = [float(sample.get('end_time', math.nan)) for sample in questions]
        if any(not math.isfinite(t) or t < 0 for t in cutoffs) or any(
                a > b for a, b in zip(cutoffs, cutoffs[1:])):
            raise ValueError(f"Question end_time must be finite, nonnegative and chronological: {video_sample['video_id']}")
        if not questions:
            return

        runtime = self.runtime_factory(self.qa_model)
        pending = deque()
        frames = None
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rekv-answer')
        snapshot = None
        stats = dict(video_id=video_sample['video_id'], max_pending_groups=self.max_pending_groups,
                     submitted_groups=0, peak_pending_groups=0, backpressure_seconds=0.)

        def collect_ready():
            # A single FIFO worker completes groups in order. Checking just the
            # head avoids scanning an arbitrarily large backlog on every frame.
            while pending and pending[0].done():
                self.records.extend(pending.popleft().result())

        try:
            producer_start = time.perf_counter()
            with runtime.encoding():
                self.start_video()
                frames = self.frame_source(video_sample)
                count = 0
                groups = groupby(zip(questions, cutoffs), key=lambda item: item[1])
                for cutoff, group in groups:
                    samples = tuple(sample for sample, _ in group)
                    for timestamp, frame in islice(frames, max(0, int(cutoff * self.sample_fps) - count)):
                        collect_ready()  # surface errors without waiting for unfinished answers
                        self.consume_frame(frame, timestamp)
                        count += 1
                        del frame
                    collect_ready()
                    # Zero means no consumer backpressure. KV publication events
                    # still protect readers; they do not wait for future writes.
                    if self.max_pending_groups and len(pending) >= self.max_pending_groups:
                        wait_start = time.perf_counter()
                        self.records.extend(pending.popleft().result())
                        stats['backpressure_seconds'] += time.perf_counter() - wait_start
                        collect_ready()
                    snapshot = runtime.capture()
                    pending.append(executor.submit(runtime.answer, self.answer_group,
                                                   video_sample, samples, snapshot))
                    snapshot = None
                    stats['submitted_groups'] += 1
                    stats['peak_pending_groups'] = max(stats['peak_pending_groups'], len(pending))
                self.finish_encoding()
            # Host-side submission completion, not a CUDA synchronization point.
            stats['producer_seconds'] = time.perf_counter() - producer_start
            drain_start = time.perf_counter()
            while pending:
                self.records.extend(pending.popleft().result())
            stats['drain_seconds'] = time.perf_counter() - drain_start
            self.async_video_stats.append(stats)
            self.finish_video(video_sample)
        finally:
            for future in pending:
                future.cancel()
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            finally:
                try:
                    runtime.close()
                finally:
                    try:
                        if frames is not None:
                            frames.close()
                    finally:
                        pending.clear()
                        snapshot = None
                        self.qa_model.clear_cache()


if __name__ == '__main__':
    work(ReKVStreamVQA, method='rekv')
