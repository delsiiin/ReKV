"""One shared model, with separate encoding and answer CUDA streams."""
import copy
from contextlib import contextmanager

import torch


def single_cuda_device(model):
    """Reject sharding and Accelerate weight offload before concurrent forwards."""
    for module in model.modules():
        for value in (getattr(module, 'hf_device_map', None) or {}).values():
            if str(value) in ('cpu', 'disk', 'meta'):
                raise ValueError('--rekv_async does not support model weight offload')
        hooks = [getattr(module, '_hf_hook', None)]
        while hooks:
            hook = hooks.pop()
            if getattr(hook, 'offload', False):
                raise ValueError('--rekv_async does not support model weight offload')
            hooks.extend(getattr(hook, 'hooks', ()))
    devices = {p.device for p in model.parameters()}
    mapped_devices = set()
    for module in model.modules():
        for value in (getattr(module, 'hf_device_map', None) or {}).values():
            device = torch.device('cuda', value) if isinstance(value, int) else torch.device(value)
            mapped_devices.add(device)
    if len(devices) != 1 or next(iter(devices)).type != 'cuda':
        raise ValueError('--rekv_async requires all model weights on one CUDA device')
    device = next(iter(devices))
    if any(d.type != 'cuda' or (d.index is not None and d.index != device.index) for d in mapped_devices):
        raise ValueError('--rekv_async does not support a sharded model')
    if model.training:
        raise ValueError('--rekv_async requires model.eval()')
    return device


class ReKVAsyncRuntime:
    def __init__(self, model):
        self.model = model
        self.device = single_cuda_device(model)
        with torch.cuda.device(self.device):
            self.encode_stream = torch.cuda.Stream(device=self.device)
            self.answer_stream = torch.cuda.Stream(device=self.device)
            # All weights and the private RoPE copy must be ready on both streams.
            self.position_bias = copy.deepcopy(model.language_model.position_bias)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
            self.encode_stream.wait_event(ready)
            self.answer_stream.wait_event(ready)

    @contextmanager
    def encoding(self):
        with torch.cuda.device(self.device), torch.cuda.stream(self.encode_stream), torch.inference_mode():
            yield

    def capture(self):
        return self.model.create_kv_snapshot(self.position_bias)

    def answer(self, callback, video_sample, sample, snapshot):
        with torch.cuda.device(self.device), torch.cuda.stream(self.answer_stream), torch.inference_mode():
            try:
                return callback(video_sample, sample, max_new_tokens=256, kv_snapshot=snapshot)
            finally:
                # Keep CPU payloads alive through H2D, including on exceptions.
                self.answer_stream.synchronize()
                snapshot = None

    def close(self):
        # Called after the answer executor has joined, before clearing the model.
        try:
            self.answer_stream.synchronize()
        finally:
            self.encode_stream.synchronize()
            for layer in self.model.kv_cache or ():
                if layer.global_stream is not None:
                    layer.global_stream.synchronize()
