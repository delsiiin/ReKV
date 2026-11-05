from importlib import import_module

from video_qa.model_paths import MODEL_PATHS
from model.video_backbone import model_family
from .common import Method, add_option, add_retrieval_arguments, retrieval_size


class ReKVMethod(Method):
    name = 'rekv'
    defaults = dict(retrieve_size=64, n_local=15000, retrieve_chunk_size=1)
    options = ('retrieve_size', 'n_local', 'retrieve_chunk_size', 'model_path')

    def add_arguments(self, parser):
        add_retrieval_arguments(parser)
        add_option(parser, '--model_path', '--model-path', default=None,
                   help='Model path or Hugging Face ID, overriding the model preset.')
        parser.add_argument('--n_local', type=int, default=None,
                            help='ReKV local attention window (default: 15000).')
        parser.add_argument('--retrieve_chunk_size', type=int, default=None)

    def validate(self, args):
        local = getattr(args, 'n_local', None)
        if local is not None and local < 1:
            raise ValueError('n_local must be positive')
        if model_family(args.model) == 'qwen2_5_vl' and local is not None and local < 196:
            raise ValueError('Qwen2.5-VL requires n_local >= 196')
        size = retrieval_size(args)
        chunk = getattr(args, 'retrieve_chunk_size', None)
        if size < 1 or (chunk is not None and not 1 <= chunk <= size):
            raise ValueError('Require 1 <= retrieve_chunk_size <= retrieve_size')

    def result_dir(self, args, dataset):
        return f'results/{args.model}/{dataset}/{retrieval_size(args)}-{args.sample_fps}'

    def load(self, args):
        modules = {
            'onevision': 'llava_onevision_rekv',
            'video_llava': 'video_llava_rekv',
            'qwen2_5_vl': 'qwen2_5_vl_rekv',
        }
        loader = import_module('model.rekv.' + modules[model_family(args.model)]).load_model
        return loader(model_path=getattr(args, 'model_path', None) or MODEL_PATHS[args.model],
                      n_local=args.n_local if args.n_local is not None else 15000,
                      topk=retrieval_size(args),
                      chunk_size=args.retrieve_chunk_size if args.retrieve_chunk_size is not None else 1)

    def analyzer_kwargs(self, args):
        return dict(retrieve_size=retrieval_size(args),
                    chunk_size=getattr(args, 'retrieve_chunk_size', None) or 1)
