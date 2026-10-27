from importlib import import_module

from video_qa.model_paths import MODEL_PATHS
from .common import Method, add_retrieval_arguments, retrieval_size


def async_pending_groups(args):
    value = getattr(args, 'rekv_async_max_pending_groups', None)
    return 2 if value is None else value


def async_result_subdir(args):
    limit = async_pending_groups(args)
    if limit == 2:
        return 'async'
    return 'async_unbounded' if limit == 0 else f'async_groups_{limit}'


class ReKVMethod(Method):
    name = 'rekv'
    defaults = dict(retrieve_size=64, n_local=15000, retrieve_chunk_size=1)
    options = ('retrieve_size', 'n_local', 'retrieve_chunk_size', 'rekv_async',
               'rekv_async_max_pending_groups')

    def add_arguments(self, parser):
        add_retrieval_arguments(parser)
        parser.add_argument('--n_local', type=int, default=None,
                            help='ReKV local attention window (default: 15000).')
        parser.add_argument('--retrieve_chunk_size', type=int, default=None)
        parser.add_argument('--rekv_async', '--rekv-async', action='store_true',
                            help='Overlap single-GPU ReKV stream encoding and QA using prefix snapshots.')
        parser.add_argument('--rekv_async_max_pending_groups', '--rekv-async-max-pending-groups',
                            type=int, default=None,
                            help='Outstanding prefix groups in async mode (default: 2; 0 disables backpressure).')

    def validate(self, args):
        limit = getattr(args, 'rekv_async_max_pending_groups', None)
        if limit is not None:
            if limit < 0:
                raise ValueError('rekv_async_max_pending_groups must be nonnegative')
            if not getattr(args, 'rekv_async', False):
                raise ValueError('--rekv_async_max_pending_groups requires --rekv_async')
        if getattr(args, 'rekv_async', False):
            if getattr(args, 'model', None) == 'llava_ov_72b':
                raise ValueError('--rekv_async does not support the multi-GPU 72B model')
            dataset = getattr(args, 'dataset', None)
            if dataset is not None and dataset not in ('rvs_ego', 'rvs_movie'):
                raise ValueError('--rekv_async is only supported for streaming datasets')
        local = getattr(args, 'n_local', None)
        if local is not None and local < 1:
            raise ValueError('n_local must be positive')
        size = retrieval_size(args)
        chunk = getattr(args, 'retrieve_chunk_size', None)
        if size < 1 or (chunk is not None and not 1 <= chunk <= size):
            raise ValueError('Require 1 <= retrieve_chunk_size <= retrieve_size')

    def result_dir(self, args, dataset):
        return f'results/{args.model}/{dataset}/{retrieval_size(args)}-{args.sample_fps}'

    def load(self, args):
        modules = {
            'llava_ov_0.5b': 'llava_onevision_rekv',
            'llava_ov_7b': 'llava_onevision_rekv',
            'llava_ov_72b': 'llava_onevision_rekv',
            'video_llava_7b': 'video_llava_rekv',
            'longva_7b': 'longva_rekv',
        }
        loader = import_module('model.rekv.' + modules[args.model]).load_model
        return loader(model_path=MODEL_PATHS[args.model],
                      n_local=args.n_local if args.n_local is not None else 15000,
                      topk=retrieval_size(args),
                      chunk_size=args.retrieve_chunk_size if args.retrieve_chunk_size is not None else 1)

    def analyzer_kwargs(self, args):
        options = dict(retrieve_size=retrieval_size(args),
                       chunk_size=getattr(args, 'retrieve_chunk_size', None) or 1)
        if getattr(args, 'rekv_async', False):
            options['rekv_async'] = True
            options['rekv_async_max_pending_groups'] = async_pending_groups(args)
        return options
