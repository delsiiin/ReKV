import math
from itertools import islice

import torch
from logzero import logger

from video_qa.base import BaseVQA, work
from video_qa.sampling import sampled_frames


class ReKVStreamVQA(BaseVQA):
    def video_open_qa(self, question, max_new_tokens=1024):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=max_new_tokens)

        return {
            'pred_answer': pred_answer.replace('\n', ''),
        }

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        questions = video_sample['conversations']
        cutoffs = [float(sample['end_time']) for sample in questions]
        if any(not math.isfinite(t) or t < 0 for t in cutoffs) or any(
                earlier > later for earlier, later in zip(cutoffs, cutoffs[1:])):
            raise ValueError(f"Question end_time must be finite, nonnegative and chronological: {video_sample['video_id']}")

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()

        frames = sampled_frames(video_sample['video_path'], self.sample_fps)
        try:
            encoded_count = 0
            for sample, cutoff in zip(questions, cutoffs):
                logger.debug(f'sample: {sample}')
                # Original ReKV slice: video[encoded_count:int(end_time * fps)].
                end_idx = int(cutoff * self.sample_fps)
                # Match SuPr-KV's read-one/encode-one input buffering while
                # retaining ReKV's full historical CPU KV cache.
                for _, frame in islice(frames, max(0, end_idx - encoded_count)):
                    self.qa_model.encode_video(torch.from_numpy(frame).unsqueeze(0))
                    encoded_count += 1
                    del frame

                qa_results = self.video_open_qa(sample['question'], max_new_tokens=256)
                self.record[(self.retrieve_size, self.chunk_size)].append({
                    'video_id': video_sample['video_id'],
                    'question': sample['question'],
                    'answer': sample['answer'],
                    'pred_answer': qa_results['pred_answer'],
                })
        finally:
            frames.close()
 

if __name__ == "__main__":
    work(ReKVStreamVQA)
