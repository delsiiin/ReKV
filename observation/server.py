"""Run with python -m observation.server; all data and computation stay local."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import json
import math
from pathlib import Path
import re
import threading
import time
import traceback
from urllib.parse import urlparse, unquote

from observation.data import RVSDataset, decode_frames


ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).parent / 'dist'
TERMINAL = {'complete', 'failed', 'cancelled'}


def validate_params(raw, dataset, max_frames):
    if not isinstance(raw, dict):
        raise ValueError('Expected a JSON object')
    allowed = {'sample_id', 'max_frames', 'sample_fps', 'start_time', 'mode', 'question_index', 'max_new_tokens'}
    if set(raw) - allowed:
        raise ValueError('Unknown observation parameter')
    sample_id = raw.get('sample_id')
    if not isinstance(sample_id, str):
        raise ValueError('sample_id must be a string')
    record = dataset.get(sample_id)
    frames, question = raw.get('max_frames', 8), raw.get('question_index', 0)
    if type(frames) is not int or not 1 <= frames <= max_frames:
        raise ValueError(f'max_frames must be an integer from 1 to {max_frames}')
    if type(question) is not int or not 0 <= question < len(record['conversations']):
        raise ValueError('Invalid question index')
    max_new_tokens = raw.get('max_new_tokens', 128)
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512:
        raise ValueError('max_new_tokens must be an integer from 1 to 512')
    mode = raw.get('mode', 'video')
    if mode not in ('video', 'question'):
        raise ValueError('mode must be video or question')
    fps, start = raw.get('sample_fps', 0.5), raw.get('start_time', 0)
    if type(fps) not in (int, float) or not math.isfinite(fps) or not 0 < fps <= 1:
        raise ValueError('sample_fps must be in (0, 1] for this RVS experiment')
    if type(start) not in (int, float) or not math.isfinite(start) or start < 0:
        raise ValueError('start_time must be a finite nonnegative number')
    return dict(sample_id=sample_id, max_frames=frames, sample_fps=float(fps),
                start_time=float(start), mode=mode, question_index=question, max_new_tokens=max_new_tokens)


class ObservationService:
    def __init__(self, args):
        self.args = args
        self.dataset = RVSDataset(args.data_root)
        self.cache = Path(args.cache_dir).resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs, self.events = {}, {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='observation')
        self.engine = None

    def cache_key(self, params):
        record = self.dataset.get(params['sample_id'])
        paths = [self.dataset.video_path(record), self.dataset.annotations[params['sample_id']]]
        model_path = Path(self.args.model_path).resolve()
        if not (model_path / 'config.json').is_file():
            raise ValueError(f'Local model config not found: {model_path}')
        paths += sorted(p for p in model_path.iterdir() if p.is_file())
        signature = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
        code = hashlib.sha256()
        for p in sorted(Path(__file__).parent.glob('*.py')):
            code.update(p.read_bytes())
        versions = {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'numpy']}
        value = dict(params=params, files=signature, code=code.hexdigest(), versions=versions,
                     device=self.args.device, dtype=self.args.dtype)
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]

    def submit(self, raw):
        params = validate_params(raw, self.dataset, self.args.max_frames)
        job_id = self.cache_key(params)
        with self.lock:
            if job_id in self.jobs and self.jobs[job_id]['status'] not in {'failed', 'cancelled'}:
                existing = copy.deepcopy(self.jobs[job_id])
                if existing['status'] == 'complete':
                    existing.update(cached=True, message='已读取本地缓存')
                return existing
            output = self.cache / job_id / 'result.json'
            if output.is_file():
                job = dict(id=job_id, status='complete', progress=1, message='已读取本地缓存',
                           cached=True, result_url=f'/artifacts/{job_id}/result.json', params=params)
                self.jobs[job_id] = job
                return copy.deepcopy(job)
            pending = sum(j['status'] not in TERMINAL for j in self.jobs.values())
            if pending >= 3:
                raise ValueError('已有 3 个计算任务；请等待完成或取消旧任务')
            job = dict(id=job_id, status='queued', progress=0, message='等待 GPU 计算',
                       params=params, cached=False)
            self.jobs[job_id] = job
            self.events[job_id] = threading.Event()
            self.pool.submit(self.run, job_id, params)
            return copy.deepcopy(job)

    def update(self, job_id, **values):
        with self.lock:
            self.jobs[job_id].update(values)

    def get_job(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError('Unknown job')
            return copy.deepcopy(self.jobs[job_id])

    def cancel(self, job_id):
        with self.lock:
            job = self.get_job(job_id)
            if job['status'] not in TERMINAL:
                self.events[job_id].set()
                self.update(job_id, message='正在取消；将在当前计算阶段结束后停止')
            return self.get_job(job_id)

    def run(self, job_id, params):
        started = time.monotonic()
        try:
            from observation.engine import Cancelled, ObservationEngine

            def check_cancel():
                if self.events[job_id].is_set():
                    raise Cancelled('任务已取消')

            def progress(stage, message, fraction):
                check_cancel()
                self.update(job_id, status='running', stage=stage, message=message, progress=fraction)

            check_cancel()
            record = copy.deepcopy(self.dataset.get(params['sample_id']))
            path = self.dataset.video_path(record)
            directory = self.cache / job_id
            progress('decoding', '读取实际采样帧', 0.03)
            images, frames, sampling = decode_frames(path, params, directory)
            for frame in frames:
                frame['image_url'] = f"/artifacts/{job_id}/{frame['image']}"
            self.update(job_id, preview=dict(frames=copy.deepcopy(frames), sampling=sampling))
            check_cancel()
            if self.engine is None:
                self.engine = ObservationEngine(self.args.model_path, self.args.device, self.args.dtype)
            result = self.engine.observe(images, frames, record, params, progress, check_cancel)
            result.update(schema_version=2, job_id=job_id, sample=record, parameters=params,
                          sampling=sampling, total_elapsed_seconds=time.monotonic() - started)
            check_cancel()
            temporary = directory / 'result.tmp'
            temporary.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            temporary.replace(directory / 'result.json')
            self.update(job_id, status='complete', progress=1, message='观察计算完成',
                        result_url=f'/artifacts/{job_id}/result.json')
        except Exception as error:
            cancelled = self.events[job_id].is_set()
            if not cancelled:
                traceback.print_exc()
            self.update(job_id, status='cancelled' if cancelled else 'failed',
                        message='任务已取消' if cancelled else f'{type(error).__name__}: {error}')
            # Do not retain the traceback (and its CUDA tensors) in the job state.
        finally:
            import gc
            gc.collect()
            if self.engine is not None:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def close(self):
        for event in self.events.values():
            event.set()
        self.pool.shutdown(wait=True, cancel_futures=True)


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, data, status=200):
            content = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(content)

        def send_file(self, path, mime):
            if not path.is_file():
                self.send_json({'error': 'File not found'}, 404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(path.stat().st_size))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            with path.open('rb') as stream:
                while chunk := stream.read(1024 * 1024):
                    self.wfile.write(chunk)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            try:
                if path == '/api/config':
                    return self.send_json(dict(model_path=str(Path(service.args.model_path).resolve()),
                                               device=service.args.device, max_frames=service.args.max_frames,
                                               data_root=str(service.dataset.root), window_size=32))
                if path == '/api/samples':
                    return self.send_json(service.dataset.summaries())
                if path.startswith('/api/samples/'):
                    return self.send_json(service.dataset.get(path.removeprefix('/api/samples/')))
                if path.startswith('/api/jobs/'):
                    return self.send_json(service.get_job(path.removeprefix('/api/jobs/')))
                match = re.fullmatch(r'/artifacts/([a-f0-9]{24})/(result\.json|frame-\d{4}\.jpg)', path)
                if match:
                    filename = match[2]
                    return self.send_file(service.cache / match[1] / filename,
                                          'application/json; charset=utf-8' if filename.endswith('.json') else 'image/jpeg')
                static = {'/': ('index.html', 'text/html; charset=utf-8'),
                          '/app.js': ('app.js', 'application/javascript; charset=utf-8'),
                          '/style.css': ('style.css', 'text/css; charset=utf-8')}
                if path in static:
                    filename, mime = static[path]
                    return self.send_file(STATIC / filename, mime)
                self.send_json({'error': 'Not found'}, 404)
            except (ValueError, KeyError) as error:
                self.send_json({'error': str(error)}, 404)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            try:
                origin = self.headers.get('Origin')
                if origin and urlparse(origin).netloc != self.headers.get('Host'):
                    return self.send_json({'error': 'Cross-origin requests are disabled'}, 403)
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    return self.send_json({'error': 'Expected application/json'}, 415)
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16384:
                    raise ValueError('Invalid request size')
                raw = json.loads(self.rfile.read(size))
                path = urlparse(self.path).path
                if path == '/api/jobs':
                    return self.send_json(service.submit(raw), 202)
                match = re.fullmatch(r'/api/jobs/([a-f0-9]{24})/cancel', path)
                if match:
                    return self.send_json(service.cancel(match[1]))
                self.send_json({'error': 'Not found'}, 404)
            except (ValueError, KeyError, TypeError) as error:
                self.send_json({'error': str(error)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                traceback.print_exc()
                self.send_json({'error': f'{type(error).__name__}: {error}'}, 500)

        def log_message(self, fmt, *args):
            # Polling once per second should not bury model errors in request logs.
            if '"GET /api/jobs/' not in (fmt % args):
                super().log_message(fmt, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/rvs')
    parser.add_argument('--model-path', type=Path, default=ROOT / 'model_zoo/llava-onevision-qwen2-7b-ov-hf')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'results/rvs_observation')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], default='float16')
    parser.add_argument('--max-frames', type=int, default=128)
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error('--max-frames must be positive')
    service = ObservationService(args)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(service))
    print(f'RVS attention observation: http://{args.host}:{args.port}', flush=True)
    print(f'{len(service.dataset.records)} samples | {args.device} | window=32 | cache={service.cache}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == '__main__':
    main()
