"""Reference offline evaluation: encode the complete video once before QA."""
import torch

from video_qa.base import BaseVQA, work


class StreamKVOfflineVQA(BaseVQA):
    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        video = self.load_video(video_sample['video_path'])
        if not isinstance(video, torch.Tensor):
            video = torch.from_numpy(video)
        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()
        self.qa_model.encode_video(video)
        for sample in video_sample['conversations']:
            self.record_answer(video_sample, sample)


if __name__ == '__main__':
    work(StreamKVOfflineVQA, method='streamkv')
