from importlib import import_module

from video_qa.model_paths import MODEL_PATHS
from .common import Method, add_retrieval_arguments, retrieval_size


class ReKVMethod(Method):
    name = 'rekv'
    defaults = dict(retrieve_size=64, n_local=15000, retrieve_chunk_size=1)
    options = ('retrieve_size', 'n_local', 'retrieve_chunk_size')

    def add_arguments(self, parser):
        add_retrieval_arguments(parser)
        parser.add_argument('--n_local', type=int, default=None,
                            help='ReKV local attention window (default: 15000).')
        parser.add_argument('--retrieve_chunk_size', type=int, default=None)

    def validate(self, args):
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
        return dict(retrieve_size=retrieval_size(args),
                    chunk_size=getattr(args, 'retrieve_chunk_size', None) or 1)
