"""Reference streaming evaluation: rebuild the entire prefix for each question."""
import torch

from video_qa.base import BaseVQA, work
from video_qa.streamkv_sampling import StreamKVVideoSource


class StreamKVStreamVQA(StreamKVVideoSource, BaseVQA):
    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        video = torch.from_numpy(self.load_video(video_sample['video_path']))
        for sample in video_sample['conversations']:
            # Preserve the source's float32 tensor multiplication and truncation.
            # start_time is unused in its prefix slice; existing ReKV annotations
            # that omit it are equivalent to start_time=0.
            windows = torch.tensor([sample.get('start_time', 0), sample['end_time']]) * self.sample_fps
            cutoff = windows.tolist()[-1]
            self.qa_model.clear_cache()
            self.qa_model.encode_init_prompt()
            self.qa_model.encode_video(video[:int(cutoff)])
            self.record_answer(video_sample, sample, max_new_tokens=256)


if __name__ == '__main__':
    work(StreamKVStreamVQA, method='streamkv')
