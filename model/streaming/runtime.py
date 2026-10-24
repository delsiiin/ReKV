"""ReKV-style single-process execution: load once, encode, answer, then resume."""
import gc
import math
import os
from dataclasses import dataclass, asdict

import torch


@dataclass(frozen=True)
class StreamingConfig:
    model_path: str = 'model_zoo/llava-onevision-qwen2-7b-ov-hf'
    device: str = 'cuda:0'
    # Compatibility aliases: different devices cannot share one model instance.
    encode_device: str | None = None
    qa_device: str | None = None
    budget_tokens: int = 4096
    recent_frames: int = 2
    topk: int = 64
    group_size: int = 128
    packed_attention: bool = False
    ema_alpha: float = .2
    min_segment_frames: int = 1
    max_segment_frames: int = 32
    summary_max_new_tokens: int = 192
    max_new_tokens: int = 128

    def __post_init__(self):
        aliases = [v for v in (self.encode_device, self.qa_device) if v is not None]
        def canonical(value):
            d = torch.device(value)
            return str(torch.device('cuda', 0 if d.index is None else d.index)) if d.type == 'cuda' else str(d)
        chosen = canonical(aliases[0] if aliases else self.device)
        if any(canonical(v) != chosen for v in aliases) or (aliases and self.device != 'cuda:0' and canonical(self.device) != chosen):
            raise ValueError('Single-model runtime requires one device; use --device and remove split-device options')
        object.__setattr__(self, 'device', chosen)
        if self.topk < self.recent_frames or self.recent_frames < 1:
            raise ValueError('topk must be >= recent_frames >= 1')
        if not 1 <= self.min_segment_frames <= self.max_segment_frames:
            raise ValueError('Invalid segment size limits')
        if not 0 < self.ema_alpha <= 1:
            raise ValueError('Invalid EMA alpha')
        if min(self.budget_tokens, self.summary_max_new_tokens, self.max_new_tokens) < 1:
            raise ValueError('Budgets and generation limits must be positive')
        if self.group_size < 1 or self.group_size % 8:
            raise ValueError('Invalid group size')



class StreamingRuntime:
    """One model in the calling process. Every operation finishes before returning.

    Submit frames/questions in chronological order. ask() blocks until generation
    completes; subsequent frames cannot affect that question's cache view.
    """
    def __init__(self, config=None, model_factory=None):
        self.config = config or StreamingConfig()
        if model_factory is None:
            from .onevision import OneVisionEngine
            model_factory = OneVisionEngine
        self.model = model_factory(asdict(self.config))
        self.question_id = 0
        self.closed = False
        self.last_timestamp = float('-inf')
        self.startup = dict(inference_pid=os.getpid(), model_instances=1,
                            execution_mode='sequential', device=self.config.device)

    def _ensure_open(self):
        if self.closed:
            raise RuntimeError('Runtime is closed')

    def _call(self, method, *args):
        self._ensure_open()
        try:
            return getattr(self.model, method)(*args)
        except BaseException:
            self.close()
            raise

    def append_frame(self, image, timestamp):
        self._ensure_open()
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < self.last_timestamp:
            raise ValueError('Frame timestamps must be finite and chronological')
        self._call('append_frame', image, timestamp)
        self.last_timestamp = timestamp
        return self.memory_stats()

    def close_segment(self):
        return self._call('close_segment')

    def ask(self, question, cutoff=None):
        self._ensure_open()
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Question must not be empty')
        if cutoff is not None and not math.isfinite(float(cutoff)):
            raise ValueError('Question cutoff must be finite')
        cutoff = self.last_timestamp if cutoff is None else float(cutoff)
        if cutoff < self.last_timestamp:
            raise ValueError('Question cutoff precedes encoded frames; historical snapshots are unavailable')
        self.question_id += 1
        question_id = self.question_id
        cache = self.model.cache
        snapshot = cache.acquire_snapshot(question_id, cutoff)
        def retrieved(hits):
            cache.record_activation(question_id, hits)
            cache.release_snapshot(question_id)
        try:
            return self._call('answer', snapshot, question, retrieved)
        finally:
            cache.release_snapshot(question_id)

    def reset(self):
        self._call('reset')  # Reuse weights; only the video state is cleared.
        self.last_timestamp = float('-inf')
        return self.memory_stats()

    def memory_stats(self):
        return self._call('stats')

    def close(self):
        if not self.closed:
            self.closed = True
            self.model.cache.snapshots.clear()
            self.model = None
            gc.collect()  # Release bound-forward/module cycles in this process.

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *args):
        self.close()
