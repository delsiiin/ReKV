"""InfiniPot reference profile and dispatch without importing model libraries."""
from dataclasses import asdict, fields
import math

from model.infinipot.config import InfiniPotConfig
from video_qa.model_paths import MODEL_PATHS
from .common import Method, add_option


class InfiniPotMethod(Method):
    name = 'infinipot'
    default_sample_fps = 2.
    defaults = asdict(InfiniPotConfig())
    options = (*defaults, 'model_path', 'max_new_tokens')

    def config(self, args):
        return InfiniPotConfig(**{key: default if getattr(args, key, None) is None else getattr(args, key)
                                 for key, default in self.defaults.items()})

    def add_arguments(self, parser):
        for field in fields(InfiniPotConfig):
            kwargs = dict(default=None)
            if field.type is bool:
                kwargs['action'] = 'store_true'
            else:
                kwargs['type'] = field.type
            if field.name == 'compression_method':
                kwargs['choices'] = ('infinipot-v', 'uniform', 'swa')
            add_option(parser, '--' + field.name, '--' + field.name.replace('_', '-'), **kwargs)
        add_option(parser, '--model_path', '--model-path', default=None)
        add_option(parser, '--max_new_tokens', '--max-new-tokens', type=int, default=None,
                   help='Explicit override of the source generation limit (300).')

    def validate(self, args):
        if not args.model.startswith('qwen2_5_vl_'):
            raise ValueError('infinipot supports qwen2_5_vl_3b, qwen2_5_vl_7b and qwen2_5_vl_72b')
        self.config(args)
        if not math.isfinite(args.sample_fps) or args.sample_fps <= 0:
            raise ValueError('sample_fps must be finite and positive')
        if getattr(args, 'max_new_tokens', None) is not None and args.max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')

    def result_dir(self, args, dataset):
        c = self.config(args)
        limit = getattr(args, 'max_new_tokens', None) or 300
        fps = args.sample_fps if args.sample_fps is not None else self.default_sample_fps
        setting = (f'{c.compression_method}_block{c.block_size}_keep{c.compress_frame_num}'
                   f'_frames{c.max_frames_num}_tar{c.tar_ratio:g}_query{c.query_ratio:g}'
                   f'_pool{int(c.adaptive_pooling)}_perframe{int(c.per_frame)}_gen{limit}')
        return f'results/infinipot/{args.model}/{dataset}/{setting}/fps_{fps:g}'

    def load(self, args):
        from model.infinipot.runtime import InfiniPotRuntime
        runtime = InfiniPotRuntime.load(args.model_path or MODEL_PATHS[args.model], self.config(args),
                                       max_new_tokens=getattr(args, 'max_new_tokens', None))
        return runtime, runtime.processor

    def analyzer_kwargs(self, args):
        return dict(record_metadata={'method': self.name, **asdict(self.config(args)),
                                     'max_new_tokens': getattr(args, 'max_new_tokens', None) or 300,
                                     'sample_fps': args.sample_fps})

    def close(self, model):
        model.close()
