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
from urllib.parse import urlparse, unquote, parse_qs

from observation.data import RVSDataset, decode_frames


ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).parent / 'dist'
TERMINAL = {'complete', 'failed', 'cancelled'}
DTYPES = ('float16', 'bfloat16', 'float32')


def validate_params(raw, dataset, default_dtype='float16'):
    if not isinstance(raw, dict):
        raise ValueError('Expected a JSON object')
    allowed = {'sample_id', 'dtype', 'max_frames', 'sample_fps', 'start_time', 'mode', 'question_index',
               'max_new_tokens', 'video_window_size', 'segment_enabled', 'segment_ema_alpha',
               'segment_threshold_multiplier', 'segment_min_frames', 'summary_max_new_tokens',
               'compression_budget_tokens', 'compression_topk', 'compression_recent_frames', 'compression_max_segment_frames',
               'question_indices', 'custom_questions'}
    if set(raw) - allowed:
        raise ValueError('Unknown observation parameter')
    sample_id = raw.get('sample_id')
    if not isinstance(sample_id, str):
        raise ValueError('sample_id must be a string')
    record = dataset.get(sample_id)
    question = raw.get('question_index', 0)
    dtype = raw.get('dtype', default_dtype)
    if dtype not in DTYPES:
        raise ValueError('dtype must be float16, bfloat16 or float32')
    max_frames = raw.get('max_frames')
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError('max_frames must be a positive integer or null for all sampled frames')
    if type(question) is not int or not 0 <= question < len(record['conversations']):
        raise ValueError('Invalid question index')
    max_new_tokens = raw.get('max_new_tokens', 128)
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512:
        raise ValueError('max_new_tokens must be an integer from 1 to 512')
    mode = raw.get('mode', 'video')
    if mode not in ('video', 'question', 'compression'):
        raise ValueError('mode must be video, question or compression')
    video_window = raw.get('video_window_size', 32)
    if type(video_window) is not int or video_window < 1:
        raise ValueError('video_window_size must be a positive integer')
    fps, start = raw.get('sample_fps', 0.5), raw.get('start_time', 0)
    if type(fps) not in (int, float) or not math.isfinite(fps) or not 0 < fps <= 1:
        raise ValueError('sample_fps must be in (0, 1] for this RVS experiment')
    if type(start) not in (int, float) or not math.isfinite(start) or start < 0:
        raise ValueError('start_time must be a finite nonnegative number')
    segment = dict(segment_enabled=raw.get('segment_enabled', True),
                   segment_ema_alpha=raw.get('segment_ema_alpha', .2),
                   segment_threshold_multiplier=1.0,
                   segment_min_frames=raw.get('segment_min_frames', 1),
                   summary_max_new_tokens=raw.get('summary_max_new_tokens', 192))
    if type(segment['segment_enabled']) is not bool:
        raise ValueError('segment_enabled must be a boolean')
    multiplier = raw.get('segment_threshold_multiplier', 1)
    if type(multiplier) not in (int, float) or multiplier != 1:
        raise ValueError('segment_threshold_multiplier is fixed at 1')
    for name, upper in [('segment_ema_alpha', 1)]:
        value = segment[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= upper:
            raise ValueError(f'{name} must be finite and in (0, {upper}]')
    if type(segment['segment_min_frames']) is not int or segment['segment_min_frames'] < 1:
        raise ValueError('segment_min_frames must be a positive integer')
    for name, upper in [('summary_max_new_tokens', 512)]:
        value = segment[name]
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f'{name} must be an integer from 1 to {upper}')
    compression = {}
    if mode != 'compression' and ('question_indices' in raw or 'custom_questions' in raw):
        raise ValueError('Multiple questions are supported in compression mode')
    if mode == 'compression':
        indices = raw.get('question_indices', [question])
        custom = raw.get('custom_questions', [])
        if not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < len(record['conversations']) for i in indices):
            raise ValueError('question_indices must be a list of valid question indices')
        if len(set(indices)) != len(indices):
            raise ValueError('question_indices must not contain duplicates')
        if not isinstance(custom, list) or any(not isinstance(q, str) or not q.strip() or len(q) > 2000 for q in custom):
            raise ValueError('custom_questions must contain nonempty strings of at most 2000 characters')
        if not 1 <= len(indices)+len(custom) <= 32:
            raise ValueError('请选择或输入 1–32 个问题')
        compression.update(question_indices=indices, custom_questions=[q.strip() for q in custom])
        for name, default, upper in [('compression_budget_tokens', 1024, 10000000),
                                     ('compression_topk', 64, 512), ('compression_recent_frames', 2, 128),
                                     ('compression_max_segment_frames', 32, 128)]:
            value = raw.get(name, default)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f'{name} must be an integer from 1 to {upper}')
            compression[name] = value
        if compression['compression_topk'] < compression['compression_recent_frames']:
            raise ValueError('检索帧数不能小于 recent 保护帧数')
        if compression['compression_max_segment_frames'] < segment['segment_min_frames']:
            raise ValueError('段帧数上限不能小于最少帧数')
    return dict(sample_id=sample_id, dtype=dtype, max_frames=max_frames, sample_fps=float(fps),
                start_time=float(start), mode=mode, question_index=question, max_new_tokens=max_new_tokens,
                video_window_size=video_window, **segment, **compression)


class ObservationService:
    def __init__(self, args):
        self.args = args
        self.dataset = RVSDataset(args.data_root)
        self.cache = Path(args.cache_dir).resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.similarity_lock = threading.Lock()
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
        for p in sorted(list(Path(__file__).parent.glob('*.py')) + list((ROOT/'model/streaming').glob('*.py')) + [ROOT/'model/patch.py']):
            code.update(p.read_bytes())
        versions = {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'numpy']}
        value = dict(params=params, files=signature, code=code.hexdigest(), versions=versions,
                     device=self.args.device)
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]

    def submit(self, raw):
        params = validate_params(raw, self.dataset, self.args.dtype)
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
            from observation.engine import Cancelled

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
            self.select_dtype(params['dtype'])
            if params['mode'] == 'compression':
                from observation.compression import compare
                revision = 0
                def publish_snapshot(side, point):
                    nonlocal revision
                    check_cancel()
                    revision += 1
                    # Only the latest scalar snapshot is retained in a running job.
                    self.update(job_id, compression_live=dict(revision=revision, side=side,
                                point=copy.deepcopy(point)))
                result = compare(self.engine, images, frames, record, params, progress, check_cancel,
                                 on_snapshot=publish_snapshot)
            else:
                result = self.engine.observe(images, frames, record, params, progress, check_cancel, artifact_dir=directory)
            if 'segment_experiment' in result:
                result['segment_experiment']['token_similarity']['url'] = f'/api/token-similarity/{job_id}'
                result['segment_experiment']['segment_attention']['url'] = f'/api/segment-attention/{job_id}'
            result.update(schema_version=26, job_id=job_id, sample=record, parameters=params,
                          sampling=sampling, total_elapsed_seconds=time.monotonic() - started)
            check_cancel()
            temporary = directory / 'result.tmp'
            temporary.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            temporary.replace(directory / 'result.json')
            self.update(job_id, status='complete', progress=1, message='观察计算完成',
                        result_url=f'/artifacts/{job_id}/result.json', compression_live=None)
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

    def select_dtype(self, dtype):
        """Called only by the single compute worker; release before loading a new dtype."""
        from observation.engine import ObservationEngine
        if self.engine is not None and self.engine.dtype_name == dtype:
            return
        self.engine = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.engine = ObservationEngine(self.args.model_path, self.args.device, dtype)

    def token_similarity(self, job_id, summary, frame, layer, representation='key'):
        from observation.segments import token_similarity
        if not re.fullmatch(r'[a-f0-9]{24}', job_id):
            raise ValueError('Invalid job id')
        # A completed result is the commit marker; partial/cancelled Keys are never served.
        with self.similarity_lock:
            directory = self.cache / job_id
            result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
            return token_similarity(directory, result, summary, frame, layer, representation)

    def segment_attention(self, job_id, summary, segment, frame, layer, rope='pre'):
        from observation.segments import segment_attention
        if not re.fullmatch(r'[a-f0-9]{24}', job_id):
            raise ValueError('Invalid job id')
        with self.similarity_lock:
            directory = self.cache / job_id
            result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
            return segment_attention(directory, result, summary, segment, frame, layer, rope)


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
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            try:
                if path == '/api/config':
                    return self.send_json(dict(model_path=str(Path(service.args.model_path).resolve()),
                                               device=service.args.device, dtype=service.args.dtype, dtypes=DTYPES,
                                               data_root=str(service.dataset.root), window_size=32,
                                               default_video_window_size=32))
                if path == '/api/samples':
                    return self.send_json(service.dataset.summaries())
                if path.startswith('/api/samples/'):
                    return self.send_json(service.dataset.get(path.removeprefix('/api/samples/')))
                if path.startswith('/api/jobs/'):
                    return self.send_json(service.get_job(path.removeprefix('/api/jobs/')))
                match = re.fullmatch(r'/api/segment-attention/([a-f0-9]{24})', path)
                if match:
                    query = parse_qs(parsed.query, keep_blank_values=True)
                    required = {'summary', 'segment', 'frame', 'layer'}
                    if not required <= set(query) <= required | {'rope'} or any(len(v) != 1 for v in query.values()):
                        raise ValueError('Expected summary, segment, frame, layer and optional rope exactly once')
                    return self.send_json(service.segment_attention(match[1], int(query['summary'][0]),
                        int(query['segment'][0]), int(query['frame'][0]), query['layer'][0], query.get('rope', ['pre'])[0]))
                match = re.fullmatch(r'/api/token-similarity/([a-f0-9]{24})', path)
                if match:
                    query = parse_qs(parsed.query, keep_blank_values=True)
                    required = {'summary', 'frame', 'layer'}
                    if not required <= set(query) <= required | {'representation'} or any(len(v) != 1 for v in query.values()):
                        raise ValueError('Expected summary, frame, layer and optional representation exactly once')
                    return self.send_json(service.token_similarity(
                        match[1], int(query['summary'][0]), int(query['frame'][0]), query['layer'][0],
                        query.get('representation', ['key'])[0]))
                match = re.fullmatch(r'/artifacts/([a-f0-9]{24})/(result\.json|frame-\d{4,}\.jpg)', path)
                if match:
                    filename = match[2]
                    return self.send_file(service.cache / match[1] / filename,
                                          'application/json; charset=utf-8' if filename.endswith('.json') else 'image/jpeg')
                static = {'/': ('index.html', 'text/html; charset=utf-8'),
                          '/app.js': ('app.js', 'application/javascript; charset=utf-8'),
                          '/attention.js': ('attention.js', 'application/javascript; charset=utf-8'),
                          '/compression.js': ('compression.js', 'application/javascript; charset=utf-8'),
                          '/style.css': ('style.css', 'text/css; charset=utf-8')}
                if path in static:
                    filename, mime = static[path]
                    return self.send_file(STATIC / filename, mime)
                self.send_json({'error': 'Not found'}, 404)
            except (ValueError, KeyError) as error:
                self.send_json({'error': str(error)}, 404)
            except FileNotFoundError as error:
                self.send_json({'error': str(error)}, 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                traceback.print_exc()
                self.send_json({'error': f'{type(error).__name__}: {error}'}, 500)

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
    parser.add_argument('--dtype', choices=DTYPES, default='float16', help='Default dtype selected in the web UI')
    args = parser.parse_args()
    service = ObservationService(args)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(service))
    print(f'RVS attention observation: http://{args.host}:{args.port}', flush=True)
    print(f'{len(service.dataset.records)} samples | {args.device} | default video window=32 | cache={service.cache}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == '__main__':
    main()
