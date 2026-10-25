from video_qa.model_paths import MODEL_PATHS, SUPRKV_MODELS
from .common import Method, add_option, add_retrieval_arguments, retrieval_size


def add_suprkv_arguments(parser):
    # None preserves StreamingConfig defaults; forward only explicitly specified arguments.
    add_option(parser, '--model_path', '--model-path', type=str, default=None,
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
    add_option(parser, '--max_new_tokens', '--max-new-tokens', type=int, default=None,
                        help='Answer token limit; SuPr-KV streaming defaults to 256, native uses per-task defaults.')
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
    values = dict(model_path=MODEL_PATHS[args.model], device=args.device or 'cuda:0',
                  topk=retrieval_size(args), packed_attention=args.packed_attention)
    values.update(config_overrides(args))
    return StreamingConfig(**values)


class SuPrKVMethod(Method):
    name = 'suprkv'
    defaults = dict(retrieve_size=64)
    options = ('retrieve_size', 'model_path', 'device', 'budget_tokens', 'recent_frames',
               'group_size', 'ema_alpha', 'min_segment_frames', 'max_segment_frames',
               'summary_max_new_tokens', 'max_new_tokens', 'packed_attention')

    def add_arguments(self, parser):
        add_retrieval_arguments(parser)
        add_suprkv_arguments(parser)
        add_option(parser, '--device', default=None)

    def validate(self, args):
        if args.model not in SUPRKV_MODELS:
            raise ValueError('SuPr-KV supports only LLaVA-OneVision Qwen2 models')
        if retrieval_size(args) < 1:
            raise ValueError('retrieve_size must be positive')

    def result_dir(self, args, dataset):
        budget = getattr(args, 'budget_tokens', None)
        budget = 8192 if budget is None else budget
        return (f'results/suprkv/{args.model}/{dataset}/'
                f'{retrieval_size(args)}-{args.sample_fps}/budget_{budget}')

    def devices(self, args, index):
        return str(index)

    def load(self, args):
        from model.suprkv.runtime import StreamingRuntime
        return StreamingRuntime(make_config(args)), None

    def close(self, model):
        model.close()

    def analyzer_kwargs(self, args):
        return dict(retrieve_size=retrieval_size(args), chunk_size=1)
