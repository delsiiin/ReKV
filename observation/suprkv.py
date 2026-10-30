"""Paired streaming-cache comparison for Observation."""
from contextlib import contextmanager
from dataclasses import asdict
import gc
import time

import torch

from model.suprkv.onevision import OneVisionEngine
from model.suprkv.profiling import StageProfiler
from model.suprkv.runtime import StreamingConfig
from model.suprkv.summarizer import OneVisionSummarizer, SummarizerLoadError


STAGES = [
    ('cache_init', '初始化缓存', False),
    ('vision_encoding', '视觉特征提取', False),
    ('video_encoding', '视频编码与入库', False),
    ('segmentation', '在线分段', True),
    ('summarizer_vision_encoding', '摘要模型视觉编码', True),
    ('summary_generation', '生成 segment summary', True),
    ('summary_query', '提取 summary Query', True),
    ('token_scoring', 'Token 评分与分类', True),
    ('token_pruning', '删除低分 token', True),
    ('quantization', '高分区域量化与渐进降位', True),
    ('frame_eviction', '整帧淘汰', True),
    ('segment_commit', '段提交与预算统计', True),
    ('retrieval', '问题前向与帧检索', False),
    ('cache_materialization', '读取 / 反量化 / 拼接 KV', False),
    ('answer_generation', '答案生成', False),
]


@contextmanager
def borrowed_engine(model, processor, config, profiler, check_cancel, summarizer=None):
    """Restore native forwards and all patch attributes, including on cancellation."""
    decoder = model.model.language_model
    saved = []
    for module, names in [(decoder, ('forward', '_old_forward', 'position_bias'))] + [
            (layer.self_attn, ('forward', '_old_forward', '_streaming_n_local')) for layer in decoder.layers]:
        for name in names:
            owned = name in vars(module) or name in module._modules
            saved.append((module, name, owned, getattr(module, name, None)))
    engine = None
    try:
        engine = OneVisionEngine(config, model=model, processor=processor,
                                 profiler=profiler, check_cancel=check_cancel, summarizer=summarizer)
        yield engine
    finally:
        if engine is not None:
            engine.cache = None
            engine.clear_segment()
            engine.summarizer = None
            engine.main_summarizer = None
            engine.segmenter.previous = None
        for module, name, owned, value in reversed(saved):
            if owned:
                setattr(module, name, value)
            elif name in vars(module) or name in module._modules:
                delattr(module, name)
        gc.collect()


def cuda_memory(device):
    if device.type != 'cuda':
        return dict(allocated_bytes=None, reserved_bytes=None, peak_allocated_bytes=None)
    torch.cuda.synchronize(device)
    return dict(allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device))


def run_pass(model, processor, images, frames, questions, config, compressed, progress, check_cancel,
             on_snapshot=None, question_progress=None):
    summarizer = None
    load_seconds = 0.
    try:
        if compressed and config.get('scoring_strategy', 'summary') == 'summary' and config.get('summarizer_model_path'):
            check_cancel()
            began = time.perf_counter()
            device = model.model.language_model.get_input_embeddings().weight.device
            summarizer = OneVisionSummarizer.load(config['summarizer_model_path'], device, model.dtype)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            load_seconds = time.perf_counter()-began
            check_cancel()
        result = _run_pass(model, processor, images, frames, questions, config, compressed,
                           progress, check_cancel, on_snapshot, question_progress, summarizer)
        result['summarizer_load_seconds'] = load_seconds
        return result
    finally:
        summarizer = None
        gc.collect()


def _run_pass(model, processor, images, frames, questions, config, compressed, progress, check_cancel,
              on_snapshot, question_progress, summarizer):
    device = model.model.language_model.get_input_embeddings().weight.device
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    initial_memory = cuda_memory(device)
    profiler = StageProfiler(device)
    began = time.perf_counter()
    with borrowed_engine(model, processor, dict(config, compression_enabled=compressed), profiler, check_cancel, summarizer) as engine:
        timeline, commits = [], []
        last_commit = None
        reporting_seconds = 0.
        segment_ids = {}
        def sample(index, kind):
            nonlocal last_commit, reporting_seconds
            reporting_started = time.perf_counter()
            stats = engine.cache.memory_stats()
            segment_ids.update({fid: frame.segment_id for fid, frame in engine.cache.frames.items()})
            frame_states = engine.cache.frame_states(segment_ids)
            if not compressed:
                # Baseline closes frames for storage bookkeeping, not semantic segmentation.
                for frame in frame_states['frames']:
                    frame['segment_id'] = None
            timeline.append(dict(frame_index=index, timestamp=frames[index]['timestamp'], kind=kind,
                                 persistent_tensor_bytes=stats['persistent_tensor_bytes'],
                                 persistent_cuda_bytes=stats['persistent_cuda_bytes'],
                                 managed_bytes=stats['persistent_bytes'],
                                 retained_frames=stats['retained_frames'],
                                 frame_states=frame_states))
            event = stats['last_event']
            if event and event['version'] != last_commit:
                commits.append(dict(event))
                last_commit = event['version']
            if on_snapshot is not None:
                on_snapshot(timeline[-1])
            reporting_seconds += time.perf_counter()-reporting_started
        for index, (image, frame) in enumerate(zip(images, frames)):
            check_cancel()
            progress(index, len(frames))
            engine.append_frame(image, frame['timestamp'])
            sample(index, 'frame')
        check_cancel()
        engine.close_segment()
        sample(len(frames)-1, 'flush')
        profiler.synchronize()
        encoding_seconds = time.perf_counter()-began-reporting_seconds
        memory_before_answer = cuda_memory(device)
        cache_stats = engine.cache.memory_stats()
        answers = []
        for question in questions:
            check_cancel()
            question_id = question['question_id']
            if question_progress is not None:
                question_progress(question_id, len(questions))
            snapshot = engine.cache.acquire_snapshot(question_id, frames[-1]['timestamp'])
            def retrieved(hits):
                engine.cache.record_activation(question_id, hits)
                engine.cache.release_snapshot(question_id)
            try:
                answer = engine.answer(snapshot, question['question'], retrieved)
                answers.append(dict(answer, **question))
            finally:
                engine.cache.release_snapshot(question_id)
                del snapshot
        profiler.synchronize()
        end_memory = cuda_memory(device)
        return dict(status='complete', answer=answers[0], answers=answers, timeline=timeline, commits=commits,
                    cache=cache_stats, model_stats=engine.stats(), timings=profiler.result(), encoding_seconds=encoding_seconds,
                    total_seconds=time.perf_counter()-began-reporting_seconds, reporting_seconds=reporting_seconds,
                    initial_cuda=initial_memory,
                    cuda_after_encoding=memory_before_answer, cuda_after_answer=end_memory)


def retrieval_statistics(run, frames, segment_ids):
    """Count each frame/segment at most once per question, across decoder layers."""
    frame_hits = {i: 0 for i in range(len(frames))}
    segment_frames = {}
    for frame_id, segment_id in sorted(segment_ids.items()):
        segment_frames.setdefault(segment_id, []).append(frame_id)
    segment_hits = dict.fromkeys(segment_frames, 0)
    for answer in run['answers']:
        selected = set(answer['hit_frames'])
        hit_segments = {segment_ids[i] for i in selected if i in segment_ids}
        answer['hit_segments'] = sorted(hit_segments)
        for frame_id in selected:
            frame_hits[frame_id] += 1
        for segment_id in hit_segments:
            segment_hits[segment_id] += 1
    return dict(question_count=len(run['answers']),
                frame_hits=[dict(frame_id=i, timestamp=frame['timestamp'], segment_id=segment_ids.get(i),
                                 hits=frame_hits[i]) for i, frame in enumerate(frames)],
                segment_hits=[dict(segment_id=i, frame_ids=segment_frames[i], hits=segment_hits[i])
                              for i in sorted(segment_frames)])


def compare(observation, images, frames, record, params, progress, check_cancel, on_snapshot=None):
    observation.load(progress)
    dtype = str(observation.model.dtype).removeprefix('torch.')
    if dtype not in ('float16', 'bfloat16', 'float32'):
        raise ValueError('压缩对照支持 float16、bfloat16 和 float32')
    if not frames or len(images) != len(frames):
        raise ValueError('Compression comparison requires matching nonempty frames/images')
    config = asdict(StreamingConfig(model_path=observation.model_path, device=observation.device,
        scoring_strategy=params.get('scoring_strategy', 'summary'),
        summarizer_model_path=params.get('summarizer_model_path'),
        budget_tokens=params['compression_budget_tokens'], topk=params['compression_topk'],
        recent_frames=params['compression_recent_frames'], max_segment_frames=params['compression_max_segment_frames'],
        min_segment_frames=params['segment_min_frames'], ema_alpha=params['segment_ema_alpha'],
        summary_max_new_tokens=params['summary_max_new_tokens'], max_new_tokens=params['max_new_tokens']))
    questions = [dict(question_index=i, question=record['conversations'][i]['question'])
                 for i in params.get('question_indices', [params['question_index']])]
    questions += [dict(question_index=None, question=q) for q in params.get('custom_questions', [])]
    questions = [dict(q, question_id=i+1) for i, q in enumerate(questions)]
    runs = {}
    for index, (name, compressed, label) in enumerate([
            ('before', False, f'压缩前 · {dtype}'), ('after', True, '压缩后 · 预算约束')]):
        check_cancel()
        def report(frame, total):
            progress('compression_'+name, f'{label}：处理帧 {frame+1}/{total}', .15+.4*index+.3*frame/total)
        def report_question(number, total):
            progress('compression_'+name, f'{label}：回答问题 {number}/{total}', .45+.4*index+.1*(number-1)/total)
        try:
            if compressed and config['summarizer_model_path']:
                progress('loading_summarizer', '加载独立摘要模型', .55)
            runs[name] = run_pass(observation.model, observation.processor, images, frames, questions,
                                 config, compressed, report, check_cancel,
                                 (lambda point: on_snapshot(name, point)) if on_snapshot is not None else None,
                                 report_question)
        except (torch.cuda.OutOfMemoryError, SummarizerLoadError):
            raise
        except (RuntimeError, ValueError) as error:
            runs[name] = dict(status='failed', error=f'{type(error).__name__}: {error}',
                             model_stats=dict(model_instances=1 + bool(compressed and config['summarizer_model_path'])))
        check_cancel()
    before, after = runs['before'], runs['after']
    # Share the compressed run's semantic segments; the baseline does not segment.
    segment_ids = {frame['frame_id']: frame['segment_id'] for point in after.get('timeline', [])
                   for frame in point['frame_states']['frames'] if frame['segment_id'] is not None}
    for run in runs.values():
        if run['status'] == 'complete':
            run['retrieval_stats'] = retrieval_statistics(run, frames, segment_ids)
    complete = before['status'] == after['status'] == 'complete'
    def difference(first, second):
        if first is None or second is None:
            return dict(before_bytes=first, after_bytes=second, saved_bytes=None, saved_fraction=None)
        return dict(before_bytes=first, after_bytes=second, saved_bytes=first-second,
                    saved_fraction=(first-second)/first if first else None)
    memory = {}
    if complete:
        memory['persistent_kv'] = difference(before['cache']['persistent_tensor_bytes'], after['cache']['persistent_tensor_bytes'])
        memory['cuda_allocated'] = difference(before['cuda_after_encoding']['allocated_bytes'], after['cuda_after_encoding']['allocated_bytes'])
        memory['cuda_peak'] = difference(before['cuda_after_answer']['peak_allocated_bytes'], after['cuda_after_answer']['peak_allocated_bytes'])
    stages = []
    for key, label, algorithm in STAGES:
        left = before.get('timings', {}).get(key, {}).get('seconds', 0.) if before['status']=='complete' else None
        right = after.get('timings', {}).get(key, {}).get('seconds', 0.) if after['status']=='complete' else None
        stages.append(dict(id=key, label=label, algorithm=algorithm, before_seconds=left, after_seconds=right,
                           enabled=config['scoring_strategy'] == 'summary' or key not in
                               ('summarizer_vision_encoding', 'summary_generation', 'summary_query'),
                           delta_seconds=right-left if left is not None and right is not None else None))
    progress('compression_complete', '压缩对照完成' if complete else '对照部分失败；查看各侧状态', .99)
    return dict(frames=frames, compression_experiment=dict(status='complete' if complete else 'partial',
        question=questions[0]['question'], question_index=questions[0]['question_index'], questions=questions,
        frame_count=len(frames),
        cutoff_timestamp=frames[-1]['timestamp'], config=config, runs=runs, memory=memory, stages=stages,
        method=dict(model_instances=after.get('model_stats', {}).get('model_instances', 1),
            scoring_strategy=config['scoring_strategy'], compression_score='pre_softmax',
            summarizer_model_path=config['summarizer_model_path'],
            execution='sequential', cache_device=observation.device, dtype=dtype,
            offload=False, baseline=f'same streaming encoding and top-k retrieval; all {dtype}; no compression or summaries',
            timing='exclusive wall time, CUDA synchronized at stage boundaries; excludes model loading and video decoding',
            memory='unique persistent tensor storage; CUDA allocated/peak include weights and workspaces; reserved allocator memory is not savings',
            frame_states='each progress point records per-frame, per-layer token counts [model dtype, INT8, INT4, INT2]; bit_order 16 denotes unquantized model dtype; evicted frames retain observed segment_id and have counts=null; baseline segment_id=null; no future frames; sink excluded',
            retrieval_hits='once per question per frame across layers, including recent frames; segment hit means any member frame was retrieved; both sides use compressed-run segments; unavailable if compressed run failed',
            sampling='identical selected frames and questions; questions are answered independently in order at the last selected frame using the same video cache; reference answers are not inputs',
            measurement='single paired run, baseline first; first-use kernel initialization can affect timing')))
