"""Paired evaluation of an EncodingPrefix optimization using real video frames.

Load a saved pre-change encoding.py as the reference; keep the same model,
attention, compression policy and input frames. Measure three alternating pairs
by default. Check every frame's decoder output, final cache payload and answers.
No external answer grading or downloads are performed.
"""
import argparse
from dataclasses import fields, is_dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from decord import VideoReader, cpu

from observation.engine import ObservationEngine
from observation.suprkv import borrowed_engine
from model.suprkv import onevision
from model.suprkv.encoding import EncodingPrefix
from model.suprkv.profiling import StageProfiler


def digest(value):
    """Hash values, not object/storage identities. Called outside timed spans."""
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            h.update(str((tuple(item.shape), item.dtype)).encode())
            h.update(item.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        elif is_dataclass(item):
            h.update(type(item).__name__.encode())
            for field in fields(item):
                h.update(field.name.encode()); visit(getattr(item, field.name))
        elif isinstance(item, dict):
            for key in sorted(item):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            h.update(repr(item).encode())
        h.update(b'\0')
    visit(value)
    return h.hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-result', required=True)
    parser.add_argument('--reference-source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    torch.set_num_threads(1)
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('model.suprkv._reference_prefix', args.reference_source)
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    document = json.loads(Path(args.source_result).read_text())
    experiment = document['compression_experiment']
    config = dict(experiment['config'], device=args.device, compression_enabled=True)
    frames = document['frames']
    print('Decoding original frames', flush=True)
    video = VideoReader(document['sample']['video_path'], ctx=cpu(0), num_threads=1)
    images = video.get_batch([frame['source_index'] for frame in frames]).asnumpy()
    del video
    observation = ObservationEngine(config['model_path'], args.device, experiment['method']['dtype'])
    observation.load(lambda *a: None)
    report = dict(source_result=args.source_result, reference_source=args.reference_source,
        reference_sha256=hashlib.sha256(Path(args.reference_source).read_bytes()).hexdigest(),
        optimized_sha256=hashlib.sha256(Path(onevision.__file__).with_name('encoding.py').read_bytes()).hexdigest(),
        config=config, dtype=experiment['method']['dtype'], frames=len(frames),
        device=torch.cuda.get_device_name(args.device), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        rounds=args.rounds, runs=[], checks={},
        timing='exclusive synchronized stage timers; encoding excludes input decoding, model loading, QA and correctness hashing')
    def save():
        (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    save()
    # Warm the model and normal attention path before either measured variant.
    with borrowed_engine(observation.model, observation.processor, config, None, lambda: None) as warm:
        for frame, image in zip(frames[:8], images[:8]):
            warm.append_frame(image, frame['timestamp'])
    del warm
    reference_hashes = None
    reference_state = None
    reference_answers = None
    for round_index in range(args.rounds):
        order = ('reference', 'optimized') if round_index % 2 == 0 else ('optimized', 'reference')
        for variant in order:
            cls = reference_module.EncodingPrefix if variant == 'reference' else EncodingPrefix
            with patch.object(onevision, 'EncodingPrefix', cls), borrowed_engine(
                    observation.model, observation.processor, config, StageProfiler(args.device), lambda: None) as engine:
                profiler = engine.profiler
                original_contexts = engine.encoding_prefix.contexts
                original_forward = engine.decoder.forward
                latest_hidden = None
                def contexts(*a, **kw):
                    with profiler.measure('encoding_prefix'):
                        return original_contexts(*a, **kw)
                def forward(*a, **kw):
                    nonlocal latest_hidden
                    with profiler.measure('llm_prefill'):
                        out = original_forward(*a, **kw)
                    latest_hidden = out.last_hidden_state
                    return out
                hashes, timeline = [], []
                encoding_seconds = 0.
                with patch.object(engine.encoding_prefix, 'contexts', contexts), patch.object(engine.decoder, 'forward', forward):
                    for index, (frame, image) in enumerate(zip(frames, images)):
                        torch.cuda.synchronize(); start = time.perf_counter()
                        engine.append_frame(image, frame['timestamp'])
                        torch.cuda.synchronize(); elapsed = time.perf_counter()-start
                        encoding_seconds += elapsed
                        # Transfers and hashing below are outside measured encoding.
                        hashes.append(digest(latest_hidden))
                        latest_hidden = None
                        timeline.append(dict(frame=index, seconds=elapsed,
                            compressions=engine.cache.counts['compressions'],
                            managed_bytes=engine.cache.persistent_bytes()))
                        if index % 50 == 49:
                            print(round_index, variant, index+1, round(encoding_seconds, 3), flush=True)
                    start = time.perf_counter(); engine.close_segment(); torch.cuda.synchronize()
                    encoding_seconds += time.perf_counter()-start
                payload = digest((engine.cache.sink, engine.cache.frames, engine.cache.offloaded_frames))
                run = dict(round=round_index, variant=variant, encoding_seconds=encoding_seconds,
                    timings=profiler.result(), frame_output_hashes=hashes, final_cache_hash=payload,
                    cache=engine.cache.memory_stats(), encoding_prefix_working_bytes=engine.encoding_prefix.working_bytes,
                    timeline=timeline)
                if reference_hashes is None:
                    reference_hashes, reference_state = hashes, payload
                if hashes != reference_hashes or payload != reference_state:
                    report['checks']['exact_match'] = False
                    report['runs'].append(run); save()
                    raise AssertionError('Frame outputs or final cache differ from reference')
                # Evaluate all questions in the first pair; later pairs time encoding only.
                if round_index == 0:
                    engine.profiler = engine.cache.profiler = None
                    answers = []
                    for item in experiment['questions']:
                        question_id = item['question_id']
                        snapshot = engine.cache.acquire_snapshot(question_id, frames[-1]['timestamp'])
                        def retrieved(hits):
                            engine.cache.record_activation(question_id, hits)
                            engine.cache.release_snapshot(question_id)
                        try:
                            answer = engine.answer(snapshot, item['question'], retrieved)
                            answers.append(dict(question=item['question'], answer=answer['answer'],
                                                hit_frames=answer['hit_frames'], finish_reason=answer['finish_reason']))
                        finally:
                            engine.cache.release_snapshot(question_id)
                            del snapshot
                    run['answers'] = answers
                    if reference_answers is None:
                        reference_answers = answers
                    elif answers != reference_answers:
                        report['checks']['answers_exact_match'] = False
                        report['runs'].append(run); save()
                        raise AssertionError('Answers or retrieval hits differ from reference')
                report['runs'].append(run)
                report['checks'] = dict(exact_match=True, frame_outputs_exact_match=True,
                    final_cache_exact_match=True, answers_exact_match=reference_answers is not None,
                    question_count=len(reference_answers or []))
                save()
            del engine, latest_hidden, original_contexts, original_forward
            torch.cuda.empty_cache()
    grouped = {name:[r['encoding_seconds'] for r in report['runs'] if r['variant']==name]
               for name in ('reference','optimized')}
    medians = {name:statistics.median(values) for name,values in grouped.items()}
    stage_medians = {}
    for key in report['runs'][0]['timings']:
        stage_medians[key] = {name:statistics.median([r['timings'].get(key,{}).get('seconds',0.)
            for r in report['runs'] if r['variant']==name]) for name in grouped}
    report['summary'] = dict(encoding_seconds=grouped, median_seconds=medians,
        reduction_fraction=1-medians['optimized']/medians['reference'], speedup=medians['reference']/medians['optimized'],
        stage_medians_seconds=stage_medians)
    save()
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
