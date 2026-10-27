"""Method-neutral QA prompts, records, dataset partitioning and worker setup."""
import warnings
import random
import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import logging
import logzero
from logzero import logger

from video_qa.sampling import load_samples, nonnegative_int, sampled_frames, split_samples
from video_qa.model_paths import MODEL_PATHS
from video_qa.methods import get_method, validate_arguments
from video_qa.methods.rekv import async_result_subdir


class BaseVQA:
    def __init__(self, anno, save_dir, sample_fps,
                 qa_model, qa_processor=None,
                 num_chunks=None, chunk_idx=None,
                 record_metadata=None) -> None:
        
        self.sample_fps = sample_fps

        self.qa_model = qa_model
        self.qa_processor = qa_processor

        self.record_metadata = dict(record_metadata or {})
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        if num_chunks is not None:
            anno = self.get_chunk(anno, num_chunks, chunk_idx)
        self.anno = anno
        first_question = next((q for video in anno for q in video['conversations']), {})
        self.eval_grounding = 'temporal_windows' in first_question

        self.save_dir = save_dir
        self.choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.record = {tuple(self.record_metadata.values()): []}

    @property
    def records(self):
        return self.record[tuple(self.record_metadata.values())]

    def split_list(self, lst, n):
        return split_samples(lst, n)

    def get_chunk(self, lst, n, k):
        chunks = self.split_list(lst, n)
        return chunks[k]

    def load_video(self, video_path):
        frames = [frame for _, frame in sampled_frames(video_path, self.sample_fps)]
        if not frames:
            raise ValueError(f'Video has no sampled frames: {video_path}')
        video = np.stack(frames)
        logger.debug(f'video shape: {video.shape}')
        return video
    
    def format_mcqa_prompt(self, question, candidates):
        assert len(question) > 0, f"Q: {question}"

        formatted_choices = "\n".join(["(" + self.choice_letters[i] + ") " + candidate for i, candidate in enumerate(candidates)])
        formatted_question = f"Question: {question}\nOptions:\n{formatted_choices}\nOnly give the best option."

        return {
            "question": f"{question}",
            "formatted_question": formatted_question,
            "prompt": self.qa_model.get_prompt(formatted_question, mc=True)
        }

    def extract_characters_regex(self, s):
        s = s.strip()
        if not s:
            return ''
        if ")" in s:
            index = s.index(")")
            pred = s[index - 1 : index]
            return pred
        else:
            return s[0]

    def video_open_qa(self, question, max_new_tokens=1024, **generation_options):
        inputs = {'question': question, 'prompt': self.qa_model.get_prompt(question)}
        answer = self.qa_model.question_answering(inputs, max_new_tokens=max_new_tokens, **generation_options)
        return {'pred_answer': answer.replace('\n', '')}

    def video_close_qa(self, question, candidates, correct_choice, **generation_options):
        inputs = self.format_mcqa_prompt(question, candidates)
        answer = self.qa_model.question_answering(inputs, max_new_tokens=16, **generation_options)
        letter = self.extract_characters_regex(answer)
        return {'pred_answer': answer.replace('\n', ''), 'pred_choice': letter,
                'acc': float(letter == correct_choice)}

    def record_answer(self, video_sample, sample, *, max_new_tokens=1024):
        self.records.append(self.answer_record(video_sample, sample, max_new_tokens=max_new_tokens))

    def answer_record(self, video_sample, sample, *, max_new_tokens=1024, **generation_options):
        """Return a record without publishing it (also used by background QA)."""
        question, answer = sample['question'], sample['answer']
        record = dict(video_id=video_sample['video_id'], question=question)
        if 'choices' in sample:
            choices = sample['choices']
            # Existing benchmark convention when the ground truth is private.
            if answer is None:
                answer = choices[0]
            correct = self.choice_letters[choices.index(answer)]
            result = self.video_close_qa(question, choices, correct, **generation_options)
            record.update(choices=choices, answer=answer, correct_choice=correct,
                          pred_answer=result['pred_answer'], pred_choice=result['pred_choice'],
                          qa_acc=result['acc'] * 100)
        else:
            result = self.video_open_qa(question, max_new_tokens=max_new_tokens, **generation_options)
            record.update(answer=answer, pred_answer=result['pred_answer'])
        if 'question_type' in sample:
            record['task'] = sample['question_type']
        return record

    def analyze_a_video(self, video_sample):
        raise NotImplementedError

    def analyze(self, debug=False):
        video_annos = self.anno[:1] if debug else self.anno
        for video_sample in tqdm(video_annos):
            logger.debug(f'video_id: {video_sample["video_id"]}')
            self.analyze_a_video(video_sample)

        final_df = pd.DataFrame(self.records)
        for name, value in self.record_metadata.items():
            final_df[name] = value
        final_df.to_csv(f'{self.save_dir}/{self.num_chunks}_{self.chunk_idx}.csv', index=False)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def build_parser(method='rekv'):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_fps", '--sample-fps', type=float, default=1)
    parser.add_argument("--num_chunks", '--num-chunks', type=int, default=1)
    parser.add_argument("--max_samples", '--max-samples', type=nonnegative_int, default=0,
                        help="Maximum video samples before chunking; 0 uses the full dataset.")
    parser.add_argument("--chunk_idx", '--chunk-idx', type=int, default=0)
    parser.add_argument("--save_dir", '--save-dir', type=str, required=True)
    parser.add_argument("--anno_path", '--anno-path', type=str, required=True)
    parser.add_argument("--model", choices=MODEL_PATHS, default="llava_ov_7b")
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False)
    get_method(method).add_arguments(parser)
    return parser


def work(QA_CLASS, method='rekv', argv=None):
    logging.set_verbosity_error()
    parser = build_parser(method)
    args = parser.parse_args(argv)
    implementation = get_method(method)
    try:
        validate_arguments(args, implementation)
        if getattr(args, 'rekv_async', False) and not getattr(QA_CLASS, 'supports_rekv_async', False):
            raise ValueError('--rekv_async is only supported by the ReKV streaming entry point')
    except ValueError as error:
        parser.error(str(error))

    if getattr(args, 'rekv_async', False):
        subdir = async_result_subdir(args)
        if os.path.basename(os.path.normpath(args.save_dir)) != subdir:
            args.save_dir = os.path.join(args.save_dir, subdir)

    if not args.debug:
        logzero.loglevel(logging.INFO)
        warnings.filterwarnings('ignore')

    os.makedirs(args.save_dir, exist_ok=True)

    # fix random seed
    random.seed(2024)
    logger.info('seed: 2024')

    # Select globally before chunking, and validate before loading the model.
    anno = load_samples(args.anno_path, args.max_samples)
    if not 1 <= args.num_chunks <= len(anno):
        parser.error("num_chunks must be between 1 and the number of selected samples")
    if not 0 <= args.chunk_idx < args.num_chunks:
        parser.error("chunk_idx must be between 0 and num_chunks - 1")

    videoqa_model, videoqa_processor = implementation.load(args)

    try:
        analyzer = QA_CLASS(
            anno=anno,
            sample_fps=args.sample_fps,
            qa_model=videoqa_model,
            qa_processor=videoqa_processor,
            num_chunks=args.num_chunks,
            chunk_idx=args.chunk_idx,
            save_dir=args.save_dir,
            **implementation.analyzer_kwargs(args),
        )

        analyzer.analyze(debug=args.debug)
    finally:
        implementation.close(videoqa_model)
