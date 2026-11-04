"""Run the real launcher with opt-in process/stage memory tracing.

Example:
  python tests/benchmark_suprkv_memory.py --output /tmp/suprkv-head --gpu 1 -- \
      bash run_suprkv.sh --compression-granularity head

Predictions are redirected under output/results. External answer scoring is
skipped: this diagnostic measures inference and never calls the grading API.
--max-questions explicitly limits diagnostics to a prefix of each video's QA.
--native-profiler-path enables an optional separately installed memray profiler.
No production inference implementation is modified.
"""
import argparse
import ctypes
from functools import wraps
from contextlib import nullcontext
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import threading
import time
import tracemalloc
import weakref


ROOT = Path(__file__).resolve().parents[1]


class MemoryTrace:
    def __init__(self, directory):
        self.path = Path(directory) / f'memory-{os.getpid()}.jsonl'
        self.lock = threading.Lock()
        self.engine = lambda: None
        self.stages = []
        self.stop = threading.Event()
        self.libc = ctypes.CDLL(None)
        class Mallinfo(ctypes.Structure):
            _fields_ = [(name, ctypes.c_size_t) for name in
                        ('arena', 'ordblks', 'smblks', 'hblks', 'hblkhd',
                         'usmblks', 'fsmblks', 'uordblks', 'fordblks', 'keepcost')]
        self.mallinfo = getattr(self.libc, 'mallinfo2', None)
        if self.mallinfo:
            self.mallinfo.restype = Mallinfo
        if os.environ.get('SUPRKV_PROFILE_PYTHON') == '1':
            tracemalloc.start(1)
        self.emit('worker_start')
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def emit(self, event):
        with self.lock:
            row = dict(time=time.time(), pid=os.getpid(), event=event,
                       stage=self.stages[-1] if self.stages else 'outside_engine',
                       cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
            with open('/proc/self/status') as stream:
                for line in stream:
                    parts = line.split()
                    if parts[0] in ('VmRSS:', 'VmHWM:', 'RssAnon:', 'RssFile:', 'RssShmem:', 'VmSize:'):
                        row[parts[0][:-1]] = int(parts[1]) * 1024
                    elif parts[0] == 'Threads:':
                        row['threads'] = int(parts[1])
            if self.mallinfo:
                info = self.mallinfo()
                row['malloc'] = {k: getattr(info, k) for k in ('arena', 'uordblks', 'fordblks', 'hblkhd')}
            if tracemalloc.is_tracing():
                row['python_current'], row['python_peak'] = tracemalloc.get_traced_memory()
                row['python_tracer_overhead'] = tracemalloc.get_tracemalloc_memory()
            engine = self.engine()
            cache = getattr(engine, 'cache', None)
            if cache is not None:
                row.update(frame=cache.next_id, retained=len(cache.frames),
                           granularity=cache.compression_granularity,
                           snapshots=len(cache.snapshots), budget=cache.budget)
                for label, ledger in (('persistent', cache._persistent_ledger), ('working', cache._working_ledger)):
                    row[label] = dict(nodes=len(ledger.nodes), storages=len(ledger.storages),
                                      managed_bytes=ledger.total, tensor_bytes=ledger.tensor_bytes)
            kernels = {}
            for name in ('model.suprkv.packed_kernels',
                         'model.suprkv.ragged_kernels',
                         'model.rekv.attention.dot_production_attention.triton_impl'):
                module = sys.modules.get(name)
                if module is None:
                    continue
                for key, fn in tuple(vars(module).items()):
                    if not key.startswith('_'):
                        continue
                    while hasattr(fn, 'fn') and not hasattr(fn, 'device_caches'):
                        fn = fn.fn
                    if hasattr(fn, 'device_caches'):
                        kernels[name.rsplit('.', 1)[-1]+'.'+key] = sum(
                            len(entry[0]) for entry in tuple(fn.device_caches.values()))
            row['triton_variants'] = kernels
            with self.path.open('a') as stream:
                stream.write(json.dumps(row) + '\n')

    def loop(self):
        while not self.stop.wait(2):
            self.emit('interval')

    def install(self):
        from model.suprkv import profiling
        original = profiling.stage
        trace = self
        def stage(name):
            def decorate(function):
                wrapped = original(name)(function)
                @wraps(function)
                def call(engine, *args, **kwargs):
                    trace.engine = weakref.ref(engine)
                    trace.stages.append(name)
                    cache = getattr(engine, 'cache', None)
                    record = name != 'video_encoding' or getattr(cache, 'next_id', 0) % 16 == 0
                    if record:
                        trace.emit(name + ':before')
                    try:
                        return wrapped(engine, *args, **kwargs)
                    finally:
                        if record:
                            trace.emit(name + ':after')
                        trace.stages.pop()
                return call
            return decorate
        profiling.stage = stage


def child(arguments):
    sys.path.insert(0, str(ROOT))
    if len(arguments) >= 2 and arguments[0] == '-m':
        module, args = arguments[1], arguments[2:]
        sys.argv = [module, *args]
        if module == 'video_qa.run_eval':
            from video_qa import run_eval
            original_settings, original_exec = run_eval.get_eval_settings, run_eval.exec
            def settings(*args, **kwargs):
                chunks, directory = original_settings(*args, **kwargs)
                if chunks != 1 and os.environ.get('SUPRKV_PROFILE_GPU'):
                    raise ValueError('--gpu requires a single evaluation worker')
                return chunks, str(Path(os.environ['SUPRKV_PROFILE_DIR']) / directory)
            def execute(cmd, sub=False, device=None):
                if not sub and isinstance(cmd, str) and cmd.startswith('python video_qa/eval/'):
                    print('Memory diagnostic: skipping external answer scoring.', flush=True)
                    return
                return original_exec(cmd, sub, os.environ.get('SUPRKV_PROFILE_GPU', device) if sub else device)
            run_eval.get_eval_settings, run_eval.exec = settings, execute
            run_eval.main()
            return
        if module in ('video_qa.suprkv_stream_vqa', 'video_qa.suprkv_offline_vqa'):
            trace = MemoryTrace(os.environ['SUPRKV_PROFILE_DIR'])
            try:
                trace.install()
                if os.environ.get('SUPRKV_PROFILE_LEGACY_HEAD') == '1':
                    from model.suprkv import attention
                    attention.ragged_supported = lambda query: False
                from video_qa import base
                limit = int(os.environ.get('SUPRKV_PROFILE_QUESTIONS', '0'))
                if limit:
                    load_samples = base.load_samples
                    def limited_samples(*args, **kwargs):
                        return [dict(s, conversations=s['conversations'][:limit])
                                for s in load_samples(*args, **kwargs)]
                    base.load_samples = limited_samples
                tracker = nullcontext()
                if os.environ.get('SUPRKV_PROFILE_NATIVE'):
                    sys.path.insert(0, os.environ['SUPRKV_PROFILE_NATIVE'])
                    import memray
                    tracker = memray.Tracker(str(Path(os.environ['SUPRKV_PROFILE_DIR']) / 'native.bin'),
                        native_traces=True, file_format=memray.FileFormat.AGGREGATED_ALLOCATIONS)
                with tracker:
                    runpy.run_module(module, run_name='__main__')
            finally:
                trace.stop.set()
                trace.thread.join()
                trace.emit('worker_end')
            return
    os.execv(sys.executable, [sys.executable, *arguments])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--python-allocations', action='store_true')
    parser.add_argument('--gpu', type=int, help='Physical GPU index for the single inference worker')
    parser.add_argument('--max-questions', type=int, default=0,
                        help='Diagnostic prefix of each video question list; 0 keeps all questions')
    parser.add_argument('--native-profiler-path', help='Directory containing an optional memray installation')
    parser.add_argument('--legacy-head', action='store_true',
                        help='Benchmark the retained per-head fallback instead of batched CUDA attention')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.max_questions < 0 or (args.gpu is not None and args.gpu < 0):
        parser.error('GPU index and question limit must be nonnegative')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Provide the launcher command after --')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shim = output / 'bin'
    shim.mkdir()
    executable = shim / 'python'
    executable.write_text('#!/bin/sh\nexec ' + shlex.join(
        [sys.executable, str(Path(__file__).resolve()), '--child']) + ' "$@"\n')
    executable.chmod(0o755)
    env = dict(os.environ, PATH=str(shim)+os.pathsep+os.environ['PATH'],
               SUPRKV_PROFILE_DIR=str(output), SUPRKV_PROFILE_PYTHON=str(int(args.python_allocations)),
               PYTHONUNBUFFERED='1')
    if args.gpu is not None:
        env['SUPRKV_PROFILE_GPU'] = str(args.gpu)
    env['SUPRKV_PROFILE_QUESTIONS'] = str(args.max_questions)
    env['SUPRKV_PROFILE_LEGACY_HEAD'] = str(int(args.legacy_head))
    if args.native_profiler_path:
        env['SUPRKV_PROFILE_NATIVE'] = str(Path(args.native_profiler_path).resolve())
    (output / 'run.json').write_text(json.dumps(dict(command=command, python=sys.executable,
        started=time.time(), python_allocations=args.python_allocations, gpu=args.gpu,
        max_questions=args.max_questions, native_profiler_path=args.native_profiler_path,
        legacy_head=args.legacy_head), indent=2)+'\n')
    with (output / 'run.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    (output / 'exit.json').write_text(json.dumps(dict(returncode=result.returncode, finished=time.time()))+'\n')
    print(f'Exit {result.returncode}; artifacts: {output}', flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--child']:
        child(sys.argv[2:])
    else:
        main()
