"""ReKV offline QA entry point."""
from video_qa.base import work
from video_qa.flows import OfflineVQA
from video_qa.methods.lifecycle import ReKVLifecycle
from video_qa.sampling import sampled_frames


class ReKVOfflineVQA(ReKVLifecycle, OfflineVQA):
    def frame_source(self, video_sample):
        return sampled_frames(video_sample['video_path'], self.sample_fps)


if __name__ == '__main__':
    work(ReKVOfflineVQA, method='rekv')
