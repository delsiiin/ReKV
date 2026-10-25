"""Register a method here to add loading, CLI, result layout and worker dispatch."""
from .native import NativeMethod
from .rekv import ReKVMethod
from .suprkv import SuPrKVMethod

METHODS = {method.name: method for method in (ReKVMethod(), SuPrKVMethod(), NativeMethod())}


def get_method(name):
    try:
        return METHODS[name]
    except KeyError:
        raise ValueError(f'Unknown inference method: {name}') from None


def validate_arguments(args, method):
    """Reject explicitly supplied options belonging to other methods."""
    for candidate in METHODS.values():
        for name in set(candidate.options) - set(method.options):
            value = getattr(args, name, None)
            if value is not None and value is not False:
                raise ValueError(f'--{name} is not supported by --method {method.name}')
    method.validate(args)
