"""Shared offline and streaming video traversal.

Methods implement start_video, consume_frame, finish_encoding, and finish_video.
The frame-source hook keeps sampling injectable without coupling it to a method.
"""
import math
from itertools import islice

import torch

from video_qa.base import BaseVQA
from video_qa.sampling import sampled_frames


class VideoFlow(BaseVQA):
    def frame_source(self, video_sample):
        return sampled_frames(video_sample['video_path'], self.sample_fps)

    def start_video(self):
        raise NotImplementedError

    def consume_frame(self, frame, timestamp):
        raise NotImplementedError

    def finish_encoding(self):
        pass

    def finish_video(self, video_sample):
        pass


class OfflineVQA(VideoFlow):
    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        self.start_video()
        frames = self.frame_source(video_sample)
        try:
            count = 0
            for timestamp, frame in frames:
                self.consume_frame(frame, timestamp)
                count += 1
                del frame
            if not count:
                raise ValueError(f"Video has no sampled frames: {video_sample['video_path']}")
            self.finish_encoding()
        finally:
            frames.close()
        for sample in video_sample['conversations']:
            self.record_answer(video_sample, sample)
        self.finish_video(video_sample)


class StreamVQA(VideoFlow):
    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        questions = video_sample['conversations']
        cutoffs = [float(sample.get('end_time', math.nan)) for sample in questions]
        if any(not math.isfinite(t) or t < 0 for t in cutoffs) or any(
                earlier > later for earlier, later in zip(cutoffs, cutoffs[1:])):
            raise ValueError(f"Question end_time must be finite, nonnegative and chronological: {video_sample['video_id']}")
        self.start_video()
        frames = self.frame_source(video_sample)
        try:
            count = 0
            for sample, cutoff in zip(questions, cutoffs):
                # Preserve the benchmark's sampled-array slice convention.
                end_index = int(cutoff * self.sample_fps)
                for timestamp, frame in islice(frames, max(0, end_index - count)):
                    self.consume_frame(frame, timestamp)
                    count += 1
                    del frame
                self.record_answer(video_sample, sample, max_new_tokens=256)
            self.finish_encoding()
        finally:
            frames.close()
        self.finish_video(video_sample)
