"""Optional synchronized, exclusive stage timers for local comparison runs."""
from contextlib import contextmanager, nullcontext
from functools import wraps
import time

import torch


class StageProfiler:
    def __init__(self, device):
        self.device = torch.device(device)
        self.seconds, self.calls, self.stack = {}, {}, []

    def synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name):
        self.synchronize()
        frame = [time.perf_counter(), 0.]
        self.stack.append(frame)
        try:
            yield
        finally:
            self.synchronize()
            elapsed = time.perf_counter()-frame[0]
            self.stack.pop()
            self.seconds[name] = self.seconds.get(name, 0.) + max(0., elapsed-frame[1])
            self.calls[name] = self.calls.get(name, 0)+1
            if self.stack:
                self.stack[-1][1] += elapsed

    def result(self):
        return {name: dict(seconds=seconds, calls=self.calls[name]) for name, seconds in self.seconds.items()}


def measure(profiler, name):
    return profiler.measure(name) if profiler is not None else nullcontext()


def stage(name):
    def decorate(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            with measure(getattr(self, 'profiler', None), name):
                return function(self, *args, **kwargs)
        return wrapped
    return decorate
