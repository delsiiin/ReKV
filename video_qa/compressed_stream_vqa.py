"""Single-process sequential-video replay for compressed ReKV."""
import argparse
import json
import math
from pathlib import Path

from model.streaming.runtime import StreamingConfig, StreamingRuntime


def sampled_frames(path, fps=1., max_frames=0):
    import numpy as np
    from decord import VideoReader, cpu
    if fps <= 0:
        raise ValueError('fps must be positive')
    if str(path).endswith('.npy'):
        if fps > 1:
            raise ValueError('Pre-sampled NPY videos support at most 1 FPS')
        video = np.load(path, mmap_mode='r')
        indices = np.arange(0, len(video), 1/fps).astype(int)
        for count, index in enumerate(indices):
            if max_frames and count >= max_frames:
                break
            yield float(index), np.asarray(video[index]).copy()
    else:
        video = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        native_fps = video.get_avg_fps()
        stride = max(1., native_fps/fps)
        count, position, previous = 0, 0., -1
        while int(position) < len(video) and (not max_frames or count < max_frames):
            index = int(position)
            position += stride
            if index == previous:
                continue
            previous = index
            yield index/native_fps, video[index].asnumpy()
            count += 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--question', default='Describe what is happening in the video.')
    parser.add_argument('--question-time', type=float, help='Finish answering at this timestamp before encoding subsequent frames')
    parser.add_argument('--questions', help='JSON list of {time, question}, in chronological order')
    parser.add_argument('--output', help='Write answers and memory/timing statistics as JSON')
    parser.add_argument('--fps', type=float, default=1.)
    parser.add_argument('--max-frames', type=int, default=0)
    defaults = StreamingConfig()
    for field in ('model_path', 'device', 'encode_device', 'qa_device', 'budget_tokens', 'recent_frames', 'topk',
                  'group_size', 'ema_alpha', 'min_segment_frames', 'max_segment_frames',
                  'summary_max_new_tokens', 'max_new_tokens'):
        default = getattr(defaults, field)
        parser.add_argument('--'+field.replace('_', '-'), type=str if default is None else type(default), default=default)
    args = parser.parse_args(argv)
    if args.max_frames < 0:
        parser.error('--max-frames must be nonnegative')
    config = StreamingConfig(**{name: getattr(args, name) for name in defaults.__dataclass_fields__})
    questions = json.loads(Path(args.questions).read_text()) if args.questions else (
        [dict(time=args.question_time, question=args.question)] if args.question_time is not None else [])
    if not isinstance(questions, list) or any(not isinstance(q, dict) or 'time' not in q or 'question' not in q for q in questions):
        parser.error('Questions must be a JSON list of {time, question}')
    if any(not isinstance(q['time'], (int, float)) or not math.isfinite(q['time']) for q in questions):
        parser.error('Question times must be finite numbers')
    if any(not isinstance(q['question'], str) or not q['question'].strip() for q in questions):
        parser.error('Questions must be nonempty strings')
    if any(a['time'] > b['time'] for a, b in zip(questions, questions[1:])):
        parser.error('Questions must be chronological')
    answers, commits = [], []
    last_commit = None
    def record_commit(stats):
        nonlocal last_commit
        event = stats.get('last_event', {})
        if event and event['version'] != last_commit:
            commits.append(dict(event, budget_bytes=stats['budget_bytes']))
            last_commit = event['version']
    with StreamingRuntime(config) as runtime:
        index = 0
        for timestamp, frame in sampled_frames(args.video, args.fps, args.max_frames):
            while index < len(questions) and questions[index]['time'] < timestamp:
                q = questions[index]
                answers.append(dict(**q, **runtime.ask(q['question'], q['time'])))
                index += 1
            stats = runtime.append_frame(frame, timestamp)
            record_commit(stats)
            while index < len(questions) and questions[index]['time'] == timestamp:
                q = questions[index]
                answers.append(dict(**q, **runtime.ask(q['question'], q['time'])))
                index += 1
        record_commit(runtime.close_segment())
        while index < len(questions):
            q = questions[index]
            answers.append(dict(**q, **runtime.ask(q['question'], q['time'])))
            index += 1
        if not questions:
            answers.append(dict(question=args.question, **runtime.ask(args.question)))
        result = dict(answers=answers, commits=commits, memory=runtime.memory_stats(),
                      config=config.__dict__, runtime=runtime.startup)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output+'\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
