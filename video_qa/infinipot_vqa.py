"""Existing annotations/CSV scoring with source Qwen sampling and prompts."""
import math
import re

import torch

from model.infinipot.video import load_video
from video_qa.base import BaseVQA


class InfiniPotVQA(BaseVQA):
    streaming = False

    def format_mcqa_prompt(self, question, candidates):
        letters = self.choice_letters[:len(candidates)]
        if not letters or len(candidates) > len(self.choice_letters):
            raise ValueError('Expected between 1 and 8 answer choices')
        choices = '\n'.join(f'({letter}) {candidate}' for letter, candidate in zip(letters, candidates))
        # Exact source MLVU/EgoSchema instruction for four/five choices, extended
        # to the existing CGBench/QAEgo4D choice counts.
        options = ', '.join(letters[:-1]) + ' or ' + letters[-1] if len(letters) > 1 else letters[0]
        formatted = (f'Question: {question}\nOptions:\n{choices}\n'
                     f'Respond with only the letter ({options}) of the correct option.')
        return dict(question=question, formatted_question=formatted, prompt=formatted)

    def extract_characters_regex(self, response, num_choices=4):
        # Keep source A-D/A-E extraction for four/five-choice datasets. Expanding
        # every question to A-H would mistake "Final answer: D" for choice F.
        last_letter = self.choice_letters[num_choices - 1]
        pattern = (r'[\(\ ]*[A-E][\)\ ]*' if num_choices == 5
                   else r'[\(\ \[]*([A-' + last_letter + r'])[\)\.\ \]]*')
        matches = re.findall(pattern, response.replace('Answer', ''))
        letter = matches[0].strip().strip('()') if matches else ''
        return letter if letter in self.choice_letters[:num_choices] else ''

    def video_close_qa(self, question, candidates, correct_choice, **generation_options):
        inputs = self.format_mcqa_prompt(question, candidates)
        answer = self.qa_model.question_answering(inputs, **generation_options)
        letter = self.extract_characters_regex(answer, len(candidates))
        return dict(pred_answer=answer.replace('\n', ''), pred_choice=letter,
                    acc=float(letter == correct_choice))

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        questions = video_sample['conversations']
        if self.streaming:
            cutoffs = [float(sample.get('end_time', math.nan)) for sample in questions]
            if any(not math.isfinite(t) or t < 0 for t in cutoffs) or any(
                    a > b for a, b in zip(cutoffs, cutoffs[1:])):
                raise ValueError('Question end_time must be finite, nonnegative and chronological')
        else:
            cutoffs = [None] * len(questions)
        self.qa_model.reset()
        previous_cutoff = object()
        try:
            for sample, cutoff in zip(questions, cutoffs):
                if cutoff != previous_cutoff:
                    self.qa_model.set_video(load_video(
                        video_sample['video_path'], sample_fps=self.sample_fps,
                        max_frames=self.qa_model.config.max_frames_num, end_time=cutoff))
                    previous_cutoff = cutoff
                # Every question gets a fresh DynamicCache, as in the source.
                self.record_answer(video_sample, sample)
        finally:
            self.qa_model.reset()
