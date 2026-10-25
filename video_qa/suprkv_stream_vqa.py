"""SuPrKV stream QA entry point."""
from video_qa.base import work
from video_qa.flows import StreamVQA
from video_qa.methods.lifecycle import SuPrKVLifecycle
from video_qa.sampling import sampled_frames
from video_qa.methods.suprkv import add_suprkv_arguments, config_overrides, make_config


class SuPrKVStreamVQA(SuPrKVLifecycle, StreamVQA):
    def frame_source(self, video_sample):
        return sampled_frames(video_sample['video_path'], self.sample_fps)

    def video_open_qa(self, question, max_new_tokens=256):
        # Runtime configuration owns streaming generation defaults.
        answer = self.qa_model.ask(question)['answer']
        return {'pred_answer': answer.replace('\n', '')}


if __name__ == '__main__':
    work(SuPrKVStreamVQA, method='suprkv')
