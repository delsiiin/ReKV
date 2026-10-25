"""Lightweight method configuration; model dependencies are loaded in workers only."""


def add_option(parser, *flags, **kwargs):
    """Allow methods to share an option in the launcher's combined CLI."""
    if not any(flag in parser._option_string_actions for flag in flags):
        parser.add_argument(*flags, **kwargs)


def add_retrieval_arguments(parser):
    add_option(parser, '--retrieve_size', '--retrieve-size', type=int, default=None)


def retrieval_size(args):
    value = getattr(args, 'retrieve_size', None)
    return 64 if value is None else value


class Method:
    name = None
    options = ()
    defaults = {}

    def add_arguments(self, parser):
        """Register options with None/False defaults to detect explicit CLI input."""
        raise NotImplementedError

    def result_dir(self, args, dataset):
        raise NotImplementedError

    def load(self, args):
        """Return (QA runtime, optional processor); import model code lazily here."""
        raise NotImplementedError

    def validate(self, args):
        pass

    def analyzer_kwargs(self, args):
        return {}

    def close(self, model):
        pass

    def worker_arguments(self, args):
        command = []
        for name in self.options:
            value = getattr(args, name, None)
            if value is None:
                value = self.defaults.get(name)
            if value is not None and value is not False:
                command.extend(['--' + name] if value is True else ['--' + name, str(value)])
        return command

    def devices(self, args, index):
        count = 4 if args.model == 'llava_ov_72b' else 1
        return ','.join(str(count * index + offset) for offset in range(count))

    def solver(self, mode):
        return f'video_qa.{self.name}_{mode}_vqa'
