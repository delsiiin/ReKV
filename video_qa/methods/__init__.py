"""Register a method here to add loading, CLI, result layout and worker dispatch."""
from .native import NativeMethod
from .rekv import ReKVMethod
from .suprkv import SuPrKVMethod
from .streamkv import StreamKVMethod
from .infinipot import InfiniPotMethod

METHODS = {method.name: method for method in (ReKVMethod(), SuPrKVMethod(), NativeMethod(), StreamKVMethod(), InfiniPotMethod())}


def get_method(name):
    try:
        return METHODS[name]
    except KeyError:
        raise ValueError(f'Unknown inference method: {name}') from None


def validate_arguments(args, method):
    """Reject explicitly supplied options belonging to other methods."""
    if getattr(args, 'sample_fps', None) is None:
        args.sample_fps = method.default_sample_fps
    for candidate in METHODS.values():
        for name in set(candidate.options) - set(method.options):
            value = getattr(args, name, None)
            if value is not None and (value is not False or candidate.defaults.get(name) is True):
                raise ValueError(f'--{name} is not supported by --method {method.name}')
    method.validate(args)
