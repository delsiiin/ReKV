"""Reuse the current video archive for questions sharing a sampled prefix."""
import torch

from video_qa.base import BaseVQA, work
from video_qa.streamkv_sampling import StreamKVVideoSource


class StreamKVStreamVQA(StreamKVVideoSource, BaseVQA):
    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        video = torch.from_numpy(self.load_video(video_sample['video_path']))
        encoded_frames = None
        try:
            for sample in video_sample['conversations']:
                # Preserve the source's float32 multiplication and truncation.
                # Compare actual slices: distinct timestamps can select the same
                # prefix after rounding or clamping to the loaded video length.
                windows = torch.tensor([sample.get('start_time', 0), sample['end_time']]) * self.sample_fps
                prefix = video[:int(windows.tolist()[-1])]
                if encoded_frames != len(prefix):
                    self.qa_model.clear_cache()
                    self.qa_model.encode_init_prompt()
                    self.qa_model.encode_video(prefix)
                    encoded_frames = len(prefix)
                # Retrieval creates question-local generation KV; the archive is
                # reused directly, without cloning its CPU/GPU buffers.
                self.record_answer(video_sample, sample, max_new_tokens=256)
        finally:
            # Also discard partial encoding/retrieval state when a question fails.
            self.qa_model.clear_cache()


if __name__ == '__main__':
    work(StreamKVStreamVQA, method='streamkv')
