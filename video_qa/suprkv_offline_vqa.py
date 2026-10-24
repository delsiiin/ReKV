"""Offline video QA using SuPr-KV's compressed, GPU-resident video cache."""
import torch
from logzero import logger

from video_qa.base import work
from video_qa.rekv_offline_vqa import ReKVOfflineVQA
from video_qa.sampling import sampled_frames
from video_qa.suprkv_stream_vqa import SuPrKVStreamVQA


class SuPrKVOfflineVQA(SuPrKVStreamVQA):
    """Encode the entire video once, then answer all of its questions."""

    # Keep the offline open-answer prompt and 1024-token limit owned by ReKV.
    video_open_qa = ReKVOfflineVQA.video_open_qa

    def video_close_qa(self, question, candidates, correct_choice):
        input_text = self.format_mcqa_prompt(question, candidates)
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16)
        pred_letter = self.extract_characters_regex(pred_answer) if pred_answer.strip() else ''
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        self.qa_model.reset()
        frames = sampled_frames(video_sample['video_path'], self.sample_fps)
        try:
            encoded_count = 0
            for timestamp, image in frames:
                self.qa_model.append_frame(image, timestamp)
                encoded_count += 1
            if not encoded_count:
                raise ValueError(f"Video has no sampled frames: {video_sample['video_path']}")
            # Finalize the trailing segment before any offline question retrieves KV.
            self.qa_model.close_segment()
        finally:
            frames.close()

        records = self.record[(self.retrieve_size, self.chunk_size)]
        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            question = sample['question']
            answer = sample['answer']
            if 'choices' in sample:
                choices = sample['choices']
                # Match ReKV's convention for benchmarks without public ground truth.
                if answer is None:
                    answer = choices[0]
                correct_choice = self.choice_letters[choices.index(answer)]
                qa_results = self.video_close_qa(question, choices, correct_choice)
                records.append({
                    'video_id': video_sample['video_id'],
                    'question': question,
                    'choices': choices,
                    'answer': answer,
                    'correct_choice': correct_choice,
                    'pred_answer': qa_results['pred_answer'],
                    'pred_choice': qa_results['pred_choice'],
                    'qa_acc': qa_results['acc'] * 100,
                })
            else:
                qa_results = self.video_open_qa(question)
                records.append({
                    'video_id': video_sample['video_id'],
                    'question': question,
                    'answer': answer,
                    'pred_answer': qa_results['pred_answer'],
                })
            if 'question_type' in sample:
                records[-1]['task'] = sample['question_type']

        self.video_stats.append({
            'video_id': video_sample['video_id'],
            'config': self.qa_model.config.__dict__,
            'memory': self.qa_model.memory_stats(),
        })


if __name__ == '__main__':
    work(SuPrKVOfflineVQA, method='suprkv')
