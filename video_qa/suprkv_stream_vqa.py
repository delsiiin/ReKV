"""ReKV-style streaming QA with GPU-resident, compressed KV caches."""
import json
import math
from itertools import islice

import torch

from video_qa.base import BaseVQA, work
from video_qa.model_paths import MODEL_PATHS, SUPRKV_MODELS
from video_qa.sampling import sampled_frames


class SuPrKVStreamVQA(BaseVQA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_stats = []

    def video_open_qa(self, question):
        answer = self.qa_model.ask(question)['answer']
        return {'pred_answer': answer.replace('\n', '')}

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        questions = video_sample['conversations']
        cutoffs = [float(sample.get('end_time', math.nan)) for sample in questions]
        if any(not math.isfinite(t) or t < 0 for t in cutoffs) or any(
                earlier > later for earlier, later in zip(cutoffs, cutoffs[1:])):
            raise ValueError(f"Question end_time must be finite, nonnegative and chronological: {video_sample['video_id']}")

        self.qa_model.reset()
        frames = sampled_frames(video_sample['video_path'], self.sample_fps)
        try:
            encoded_count = 0
            for sample, cutoff in zip(questions, cutoffs):
                # Same frame slice as ReKV; encode only frames added since the last question.
                end_idx = int(cutoff * self.sample_fps)
                for timestamp, image in islice(frames, max(0, end_idx - encoded_count)):
                    self.qa_model.append_frame(image, timestamp)
                    encoded_count += 1

                qa_results = self.video_open_qa(sample['question'])
                self.record[(self.retrieve_size, self.chunk_size)].append({
                    'video_id': video_sample['video_id'],
                    'question': sample['question'],
                    'answer': sample['answer'],
                    'pred_answer': qa_results['pred_answer'],
                })
            self.qa_model.close_segment()
        finally:
            frames.close()

        self.video_stats.append({
            'video_id': video_sample['video_id'],
            'config': self.qa_model.config.__dict__,
            'memory': self.qa_model.memory_stats(),
        })

    def analyze(self, debug=False):
        super().analyze(debug)  # Reuse ReKV's video loop and CSV output.
        path = f'{self.save_dir}/{self.num_chunks}_{self.chunk_idx}.stats.jsonl'
        with open(path, 'w', encoding='utf-8') as output:
            for stats in self.video_stats:
                output.write(json.dumps(stats, ensure_ascii=False) + '\n')


def add_suprkv_arguments(parser):
    # None preserves StreamingConfig defaults; forward only explicitly specified arguments.
    parser.add_argument('--model_path', '--model-path', type=str, default=None,
                        help='Model path or Hugging Face ID, overriding the preset path for --model.')
    parser.add_argument('--budget_tokens', '--budget-tokens', type=int, default=None,
                        help='Persistent KV cache budget in FP16-equivalent tokens (default: 8192).')
    parser.add_argument('--recent_frames', '--recent-frames', type=int, default=None,
                        help='Number of recent sampled frames kept at original precision, also counted toward the retrieval quota (default: 2).')
    parser.add_argument('--group_size', '--group-size', type=int, default=None,
                        help='Quantization group size: K grouped along tokens and V along channels (default: 128; must be a multiple of 8).')
    parser.add_argument('--ema_alpha', '--ema-alpha', type=float, default=None,
                        help='EMA update coefficient for inter-frame distance; larger values respond faster to recent changes (default: 0.2).')
    parser.add_argument('--min_segment_frames', '--min-segment-frames', type=int, default=None,
                        help='Minimum number of sampled frames before visual changes can trigger a segment boundary (default: 4).')
    parser.add_argument('--max_segment_frames', '--max-segment-frames', type=int, default=None,
                        help='Maximum sampled frames per segment before summarization and compression (default: 32).')
    parser.add_argument('--summary_max_new_tokens', '--summary-max-new-tokens', type=int, default=None,
                        help='Maximum tokens generated for segment summaries used to score KV importance (default: 128).')
    parser.add_argument('--max_new_tokens', '--max-new-tokens', type=int, default=None,
                        help='Maximum tokens generated for each answer (default: 256).')
    parser.add_argument('--packed_attention', '--packed-attention', action='store_true',
                        help='Enable the attention kernel that reads low-bit KV directly (disabled by default).')


def config_overrides(args):
    values = dict(model_path=args.model_path, budget_tokens=args.budget_tokens,
                  recent_frames=args.recent_frames, group_size=args.group_size,
                  ema_alpha=args.ema_alpha, min_segment_frames=args.min_segment_frames,
                  max_segment_frames=args.max_segment_frames,
                  summary_max_new_tokens=args.summary_max_new_tokens,
                  max_new_tokens=args.max_new_tokens)
    return {name: value for name, value in values.items() if value is not None}


def make_config(args):
    from model.suprkv.runtime import StreamingConfig

    if args.model not in SUPRKV_MODELS:
        raise ValueError('SuPr-KV supports only LLaVA-OneVision Qwen2 models')
    values = dict(model_path=MODEL_PATHS[args.model], device=args.device,
                  topk=args.retrieve_size, packed_attention=args.packed_attention)
    values.update(config_overrides(args))
    return StreamingConfig(**values)


if __name__ == '__main__':
    work(SuPrKVStreamVQA, method='suprkv')
