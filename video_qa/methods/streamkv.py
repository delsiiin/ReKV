"""StreamKV configuration and dispatch; no model imports in the launcher."""
import argparse
from dataclasses import asdict, fields

from model.streamkv.config import StreamKVConfig
from model.video_backbone import model_family
from video_qa.model_paths import MODEL_PATHS
from .common import Method, add_option


def boolean(value):
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Expected true or false')


class StreamKVMethod(Method):
    name = 'streamkv'
    defaults = asdict(StreamKVConfig())
    options = (*defaults, 'model_path')

    def config(self, args):
        return StreamKVConfig(**{name: getattr(args, name, None) if getattr(args, name, None) is not None else default
                                 for name, default in self.defaults.items()})

    def add_arguments(self, parser):
        for field in fields(StreamKVConfig):
            arg_type = int if field.name == 'budget_tokens' else field.type
            kwargs = dict(type=boolean if arg_type is bool else arg_type, default=None)
            if field.name == 'budget_tokens':
                kwargs['help'] = ('Historical video KV token budget per layer, including summaries; '
                                  'overrides compression_ratio and compress_temp. Rounded down to frame blocks.')
            if field.name == 'segment_mode':
                kwargs['choices'] = ('semantic', 'uniform')
            add_option(parser, '--' + field.name, '--' + field.name.replace('_', '-'), **kwargs)
        add_option(parser, '--model_path', '--model-path', default=None)

    def validate(self, args):
        if model_family(args.model) != 'onevision':
            raise ValueError('streamkv currently supports only llava_ov_0.5b, llava_ov_7b and llava_ov_72b')
        config = self.config(args)
        if config.budget_tokens is not None and config.budget_tokens < 196:
            raise ValueError('budget_tokens must fit at least one frame block (196 tokens)')
        if config.retrieve_chunk_size > config.retrieve_size:
            raise ValueError('retrieve_chunk_size must be <= retrieve_size')

    def worker_arguments(self, args):
        # Explicit false values must reach the worker: the algorithm defaults
        # to enabled summary and local retrieval, unlike store_true CLI flags.
        command = []
        for name, value in asdict(self.config(args)).items():
            if value is not None:
                command.extend(['--' + name, str(value)])
        if getattr(args, 'model_path', None):
            command.extend(['--model_path', args.model_path])
        return command

    def result_dir(self, args, dataset):
        c = self.config(args)
        # All algorithm settings participate, preventing silent run collisions.
        setting = (f'{c.segment_mode}_{c.min_segment_size}-{c.max_segment_size}_theta{c.segment_theta:g}'
                   f'_summary{int(c.use_segment_summary)}_compress{c.compression_ratio:g}-{c.compress_temp:g}'
                   f'_retrieve{c.retrieve_size}-{c.retrieve_temp:g}'
                   f'_recent{int(c.retrieve_local)}-{c.retrieve_local_size}_local{c.n_local}'
                   f'_chunk{c.retrieve_chunk_size}')
        if c.budget_tokens is not None:
            setting += f'_budget{c.budget_tokens}'
        return f'results/streamkv/source/{args.model}/{dataset}/{setting}/fps_{args.sample_fps:g}'

    def load(self, args):
        from model.streamkv.runtime import StreamKVRuntime
        runtime = StreamKVRuntime.load(getattr(args, 'model_path', None) or MODEL_PATHS[args.model], self.config(args))
        return runtime, runtime.processor

    def analyzer_kwargs(self, args):
        metadata = {'method': self.name, **asdict(self.config(args))}
        # Use the column name expected by evaluation scripts.
        metadata['chunk_size'] = metadata.pop('retrieve_chunk_size')
        return dict(record_metadata=metadata)

    def close(self, model):
        model.close()
