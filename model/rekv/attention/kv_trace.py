"""Opt-in KV transfer counters: set REKV_KV_TRACE to a JSONL output path.

Counts successful copy submissions, without synchronizing CUDA or timing DMA.
Each event counts one K/V block at one layer, not one video frame.
"""
import atexit
import json
import os
import time


class KVTrace:
    def __init__(self, path):
        self.path = path
        self.counts = dict(offload_d2h=0, load_h2d=0, load_gpu_hit=0, offload_evict=0)
        self.bytes = dict(offload_d2h=0, load_h2d=0)
        self.last_report = time.monotonic()
        self.emit("enabled")
        atexit.register(self.emit, "exit")

    def emit(self, reason):
        record = dict(time=time.time(), pid=os.getpid(), reason=reason,
                      counts=self.counts, bytes=self.bytes)
        line = json.dumps(record)
        with open(self.path, "a", encoding="utf-8") as output:
            output.write(line + "\n")
        print("[ReKV KV trace] " + line, flush=True)
        self.last_report = time.monotonic()

    def record(self, event, tensors=None):
        self.counts[event] += 1
        if event in self.bytes:
            self.bytes[event] += sum(t.numel() * t.element_size() for t in tensors)
        if self.counts[event] == 1 or time.monotonic() - self.last_report >= 10:
            self.emit(event)


_path = os.environ.get("REKV_KV_TRACE")
TRACE = KVTrace(_path) if _path else None
