"""SuPr-KV dataset options, workers and launcher using the ReKV result schema."""
import argparse
import csv
import json
import math
from itertools import islice
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from video_qa.model_paths import MODEL_PATHS, SUPRKV_MODELS
from video_qa.sampling import load_samples, nonnegative_int, sampled_frames, split_samples


# Keep argument parsing and launcher imports independent of PyTorch.
CONFIG_TYPES = {
    'model_path': str,
    'budget_tokens': int,
    'recent_frames': int,
    'group_size': int,
    'ema_alpha': float,
    'min_segment_frames': int,
    'max_segment_frames': int,
    'summary_max_new_tokens': int,
    'max_new_tokens': int,
}


# Annotation files and scoring entry points are the same as the ReKV routes.
DATASETS = {
    'mlvu': ('data/mlvu/dev_debug_mc.json', 'multiple_choice'),
    'qaego4d': ('data/qaego4d/test_mc.json', 'multiple_choice'),
    'egoschema': ('data/egoschema/full.json', 'egoschema'),
    'activitynet_qa': ('data/activitynet_qa/test.json', 'open_ended'),
    'rvs_ego': ('data/rvs/ego/ego4d_oe.json', 'open_ended'),
    'rvs_movie': ('data/rvs/movie/movienet_oe.json', 'open_ended'),
    'cgbench': ('data/cgbench/full_mc.json', 'multiple_choice'),
}
STREAM_DATASETS = {'rvs_ego', 'rvs_movie'}
RESULT_FIELDS = ['video_id', 'question', 'answer', 'pred_answer', 'retrieve_size', 'chunk_size']
CHOICE_FIELDS = ['choices', 'correct_choice', 'pred_choice', 'qa_acc']


def add_suprkv_arguments(parser):
    for name, value_type in CONFIG_TYPES.items():
        parser.add_argument('--' + name, '--' + name.replace('_', '-'),
                            dest=name, type=value_type, default=None)
    parser.add_argument('--packed_attention', '--packed-attention', action='store_true')


def config_overrides(args):
    return {name: getattr(args, name) for name in CONFIG_TYPES
            if getattr(args, name, None) is not None}


def forwarded_arguments(args):
    options = []
    for name, value in config_overrides(args).items():
        options.extend(['--' + name, str(value)])
    if args.packed_attention:
        options.append('--packed_attention')
    return options


def make_config(args, *, max_new_tokens=None):
    from model.suprkv.runtime import StreamingConfig

    if args.model not in SUPRKV_MODELS:
        raise ValueError('SuPr-KV supports only LLaVA-OneVision Qwen2 models')
    values = dict(model_path=MODEL_PATHS[args.model], device=args.device,
                  topk=args.retrieve_size, packed_attention=args.packed_attention)
    if max_new_tokens is not None:
        values['max_new_tokens'] = max_new_tokens
    values.update(config_overrides(args))
    return StreamingConfig(**values)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def validate_sample(sample, streaming):
    previous = -math.inf
    for question in sample['conversations']:
        if not isinstance(question['question'], str) or not question['question'].strip():
            raise ValueError(f"Empty question in video {sample['video_id']}")
        if streaming:
            cutoff = float(question['end_time'])
            if not math.isfinite(cutoff) or cutoff < 0 or cutoff < previous:
                raise ValueError(f"Question end_time must be finite, nonnegative and chronological: {sample['video_id']}")
            previous = cutoff
        if 'choices' in question:
            choices = question['choices']
            if not 1 <= len(choices) <= 8:
                raise ValueError('Expected 1 to 8 answer choices')
            if question['answer'] is not None and question['answer'] not in choices:
                raise ValueError('Ground-truth answer is missing from choices')


def answer_row(runtime, sample, question, retrieve_size, cutoff=None):
    prompt = question['question']
    choices = question.get('choices')
    if choices is not None:
        options = '\n'.join(f'({chr(65+i)}) {choice}' for i, choice in enumerate(choices))
        prompt = f'Question: {prompt}\nOptions:\n{options}\nOnly give the best option.'
    answer = runtime.ask(prompt, cutoff)
    prediction = answer['answer'].replace('\n', '')
    row = dict(video_id=sample['video_id'], question=question['question'],
               answer=question['answer'], pred_answer=prediction,
               retrieve_size=retrieve_size, chunk_size=1)
    if choices is not None:
        # Keep unknown ground truth unknown (e.g. EgoSchema test annotations).
        correct = chr(65 + choices.index(question['answer'])) if question['answer'] is not None else ''
        match = re.search(r'\b([A-H])\b', prediction)
        predicted = match.group(1) if match else ''
        row.update(choices=choices, correct_choice=correct, pred_choice=predicted,
                   qa_acc=100. * (predicted == correct) if correct else '')
    if 'question_type' in question:
        row['task'] = question['question_type']
    return row


def evaluate_sample(runtime, sample, sample_fps, retrieve_size, streaming, frame_source):
    """Replay one video with the original ReKV sampled-array QA boundaries."""
    validate_sample(sample, streaming)
    runtime.reset()
    frames = iter(frame_source(sample['video_path'], sample_fps))
    rows = []
    try:
        if streaming:
            encoded_count = 0
            for question in sample['conversations']:
                end_idx = int(float(question['end_time']) * sample_fps)
                for timestamp, image in islice(frames, max(0, end_idx - encoded_count)):
                    runtime.append_frame(image, timestamp)
                    encoded_count += 1
                # The array slice defines visibility. Use all encoded frames;
                # their source timestamps can exceed the annotation end_time.
                rows.append(answer_row(runtime, sample, question, retrieve_size))
            runtime.close_segment()
        else:
            for timestamp, image in frames:
                runtime.append_frame(image, timestamp)
            runtime.close_segment()
            for question in sample['conversations']:
                rows.append(answer_row(runtime, sample, question, retrieve_size))
    finally:
        if hasattr(frames, 'close'):
            frames.close()
    return rows


def build_worker_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--anno_path', '--anno-path')
    parser.add_argument('--save_dir', '--save-dir', required=True)
    parser.add_argument('--model', choices=SUPRKV_MODELS, default='llava_ov_7b')
    parser.add_argument('--sample_fps', '--sample-fps', type=float, default=1.)
    parser.add_argument('--retrieve_size', '--retrieve-size', type=int, default=64)
    parser.add_argument('--max_samples', '--max-samples', type=nonnegative_int, default=0)
    parser.add_argument('--num_chunks', '--num-chunks', type=int, default=1)
    parser.add_argument('--chunk_idx', '--chunk-idx', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=False)
    add_suprkv_arguments(parser)
    return parser


def run_worker(args, runtime_factory=None, frame_source=None):
    streaming = args.dataset in STREAM_DATASETS
    if not math.isfinite(args.sample_fps) or args.sample_fps <= 0:
        raise ValueError('sample_fps must be finite and positive')
    samples = load_samples(args.anno_path or DATASETS[args.dataset][0], args.max_samples)
    chunks = split_samples(samples, args.num_chunks)
    if not 0 <= args.chunk_idx < args.num_chunks:
        raise ValueError('chunk_idx must be between 0 and num_chunks - 1')
    selected = chunks[args.chunk_idx]
    if args.debug:
        selected = selected[:1]
    for sample in selected:
        validate_sample(sample, streaming)
    default_generation = 256 if streaming else (1024 if DATASETS[args.dataset][1] == 'open_ended' else 16)
    config = make_config(args, max_new_tokens=default_generation)
    if runtime_factory is None:
        from model.suprkv.runtime import StreamingRuntime
        runtime_factory = StreamingRuntime
    if frame_source is None:
        frame_source = sampled_frames
    fields = RESULT_FIELDS + (CHOICE_FIELDS if DATASETS[args.dataset][1] != 'open_ended' else []) + ['task']
    directory = Path(args.save_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f'{args.num_chunks}_{args.chunk_idx}'
    # One model per worker; reset only video/cache state between records.
    with runtime_factory(config) as runtime, \
            (directory / f'{stem}.csv').open('w', newline='', encoding='utf-8') as output, \
            (directory / f'{stem}.stats.jsonl').open('w', encoding='utf-8') as stats:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for index, sample in enumerate(selected):
            print(f"[{args.chunk_idx + 1}/{args.num_chunks}] video {index + 1}/{len(selected)}: {sample['video_id']}", flush=True)
            writer.writerows(evaluate_sample(runtime, sample, args.sample_fps,
                                            args.retrieve_size, streaming, frame_source))
            output.flush()
            stats.write(json.dumps(dict(video_id=sample['video_id'], config=config.__dict__,
                                        memory=runtime.memory_stats()), ensure_ascii=False) + '\n')
            stats.flush()


def merge_results(directory, count):
    """Parse CSV records so multiline questions/choices survive chunk merging."""
    directory = Path(directory)
    temporary = directory / 'results.csv.tmp'
    try:
        with temporary.open('w', newline='', encoding='utf-8') as output:
            writer = None
            for index in range(count):
                with (directory / f'{count}_{index}.csv').open(newline='', encoding='utf-8') as part:
                    reader = csv.DictReader(part)
                    if writer is None:
                        writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
                        writer.writeheader()
                    elif reader.fieldnames != writer.fieldnames:
                        raise ValueError('Evaluation chunks have different CSV schemas')
                    writer.writerows(reader)
        temporary.replace(directory / 'results.csv')
    finally:
        temporary.unlink(missing_ok=True)


def launch_evaluation(args):
    from video_qa.run_eval import get_eval_settings

    annotation = args.anno_path or DATASETS[args.dataset][0]
    count, default_directory = get_eval_settings(args, args.dataset, annotation)
    directory = args.save_dir or default_directory
    if not args.only_eval:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        devices = visible.split(',') if visible is not None else [str(i) for i in range(count)]
        devices = [device.strip() for device in devices if device.strip() and device.strip() != '-1']
        if len(devices) < count:
            raise ValueError(f'num_chunks={count} needs {count} visible GPUs; got {len(devices)}')
        processes = []
        try:
            for index in range(count):
                command = [sys.executable, '-m', 'video_qa.suprkv_stream_vqa',
                           '--dataset', args.dataset, '--anno_path', annotation,
                           '--save_dir', directory, '--model', args.model,
                           '--sample_fps', str(args.sample_fps), '--retrieve_size', str(args.retrieve_size),
                           '--num_chunks', str(count), '--chunk_idx', str(index),
                           '--max_samples', str(args.max_samples), '--debug', str(args.debug)]
                command.extend(forwarded_arguments(args))
                environment = os.environ.copy()
                environment['CUDA_VISIBLE_DEVICES'] = devices[index]
                processes.append(subprocess.Popen(command, env=environment))
            failures = [(index, process.wait()) for index, process in enumerate(processes)]
            if any(code for _, code in failures):
                raise RuntimeError(f'SuPr-KV workers failed: {failures}; results were not merged or scored')
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                process.wait()
        merge_results(directory, count)
        # The open-ended scorer resumes by filename, so old judgments must not
        # be reused after replacing predictions (including --skip_eval runs).
        scoring_cache = Path(directory) / 'tmp'
        if scoring_cache.exists():
            shutil.rmtree(scoring_cache)
    if not args.skip_eval:
        scorer = DATASETS[args.dataset][1]
        command = [sys.executable, '-m', f'video_qa.eval.eval_{scorer}']
        if scorer == 'open_ended':
            command += ['--pred_path', str(Path(directory) / 'results.csv'),
                        '--output_dir', str(Path(directory) / 'tmp'),
                        '--output_json', str(Path(directory) / 'results.json')]
        else:
            command += ['--save_dir', directory]
        subprocess.run(command, check=True)


def main(argv=None):
    parser = build_worker_parser()
    args = parser.parse_args(argv)
    try:
        run_worker(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
