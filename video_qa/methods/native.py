from video_qa.model_paths import MODEL_PATHS
from .common import Method, add_option


class NativeMethod(Method):
    name = 'native'
    options = ('model_path', 'device', 'max_new_tokens')

    def add_arguments(self, parser):
        add_option(parser, '--model_path', '--model-path', default=None,
                   help='Model path or Hugging Face ID, overriding the model preset.')
        add_option(parser, '--device', default=None)
        add_option(parser, '--max_new_tokens', '--max-new-tokens', type=int, default=None)

    def validate(self, args):
        limit = getattr(args, 'max_new_tokens', None)
        if limit is not None and limit < 1:
            raise ValueError('max_new_tokens must be positive')

    def result_dir(self, args, dataset):
        return f'results/native/{args.model}/{dataset}/fps_{args.sample_fps}'

    def load(self, args):
        from model.native.runtime import NativeRuntime
        runtime = NativeRuntime.load(args.model, args.model_path or MODEL_PATHS[args.model],
                                     device=args.device, max_new_tokens=args.max_new_tokens)
        return runtime, runtime.processor

    def close(self, model):
        model.close()
