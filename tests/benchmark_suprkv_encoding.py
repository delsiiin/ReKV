"""Read-only inference diagnostic for full-prefix SuPrKV encoding.

Replays source-video frames from an Observation result. Does not score answers,
change compression policy, or write production results. CUDA profiling is run
separately from synchronized stage timing. All model loading is offline.
"""
import argparse
from contextlib import ExitStack
from functools import wraps
import json
import inspect
import textwrap
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from decord import VideoReader, cpu

from observation.engine import ObservationEngine
from observation.suprkv import borrowed_engine
from model.suprkv.profiling import StageProfiler
from model.suprkv.quantization import PackedKV


def aggregate(events):
    rows = []
    for event in events.key_averages():
        rows.append(dict(name=event.key, calls=event.count,
                         cpu_self_ms=event.self_cpu_time_total/1000,
                         cuda_self_ms=event.self_device_time_total/1000,
                         cpu_total_ms=event.cpu_time_total/1000,
                         cuda_total_ms=event.device_time_total/1000))
    return sorted(rows, key=lambda r: r['cuda_self_ms'], reverse=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-result', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--runs', nargs='+', choices=('before', 'after'), default=['before', 'after'])
    parser.add_argument('--skip-trace', action='store_true')
    parser.add_argument('--position-ablation', action='store_true', help='Diagnostic-only removal of unused prefix-position construction; checks exact output equality.')
    args = parser.parse_args()
    if args.position_ablation:
        from model.suprkv.encoding import EncodingPrefix
        if 'ids.long() + frame.rope_start' not in inspect.getsource(EncodingPrefix.contexts):
            parser.error('Position construction is already removed; use benchmark_suprkv_prefix_optimization.py with a saved reference source.')
    torch.set_num_threads(1)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    source = Path(args.source_result)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    document = json.loads(source.read_text())
    experiment = document['compression_experiment']
    frames = document['frames']
    config = dict(experiment['config'], device=args.device)
    # Decode the exact original frame indices, not lossy artifact JPEGs.
    video = VideoReader(document['sample']['video_path'], ctx=cpu(0), num_threads=1)
    images = video.get_batch([f['source_index'] for f in frames]).asnumpy()
    del video
    observation = ObservationEngine(config['model_path'], args.device, experiment['method']['dtype'])
    observation.load(lambda *args: None)
    report = dict(source=str(source), device=torch.cuda.get_device_name(args.device),
                  gpu_selection=os.environ.get('CUDA_VISIBLE_DEVICES'), frames=len(frames), config=config,
                  dtype=experiment['method']['dtype'], runs={})
    (output/'report.json').write_text(json.dumps(report, indent=2))
    for mode in args.runs:
        compressed = mode == 'after'
        profiler = StageProfiler(args.device)
        cfg = dict(config, compression_enabled=compressed,
                   compression_granularity=config['compression_granularity'] if compressed else 'layer',
                   scoring_scope=config['scoring_scope'] if compressed else 'segment',
                   offload_enabled=config['offload_enabled'] if compressed else False)
        with borrowed_engine(observation.model, observation.processor, cfg, profiler, lambda: None) as engine:
            counts = dict(materialize_calls=0, materialize_misses=0, dequantize_calls=0)
            materialize = engine.encoding_prefix._materialize
            contexts = engine.encoding_prefix.contexts
            forward = engine.decoder.forward
            append = engine.cache.append_frame
            dequantize = PackedKV.dequantize
            def counted(frame, layer):
                counts['materialize_calls'] += 1
                cached = engine.encoding_prefix.entries.get((frame.frame_id, layer))
                counts['materialize_misses'] += int(cached is None or cached[0] is not frame.layers[layer])
                return materialize(frame, layer)
            def decoded(self, *a, **kw):
                counts['dequantize_calls'] += 1
                return dequantize(self, *a, **kw)
            def timed(name, function):
                @wraps(function)
                def call(*a, **kw):
                    with profiler.measure(name):
                        return function(*a, **kw)
                return call
            timeline = []
            torch.cuda.synchronize()
            began = time.perf_counter()
            with ExitStack() as stack:
                stack.enter_context(patch.object(engine.encoding_prefix, '_materialize', counted))
                stack.enter_context(patch.object(engine.encoding_prefix, 'contexts', timed('encoding_prefix', contexts)))
                stack.enter_context(patch.object(engine.decoder, 'forward', timed('llm_prefill', forward)))
                stack.enter_context(patch.object(engine.cache, 'append_frame', timed('frame_storage', append)))
                stack.enter_context(patch.object(PackedKV, 'dequantize', decoded))
                for index, (frame, image) in enumerate(zip(frames, images)):
                    previous = dict(profiler.seconds)
                    before_counts = dict(counts)
                    engine.append_frame(image, frame['timestamp'])
                    timeline.append(dict(frame=index, times={k:v-previous.get(k,0.) for k,v in profiler.seconds.items()},
                        counts={k:v-before_counts[k] for k,v in counts.items()},
                        compressions=engine.cache.counts['compressions'],
                        managed_bytes=engine.cache.persistent_bytes(),
                        prefix_working_bytes=engine.encoding_prefix.working_bytes))
                    if index % 25 == 24:
                        print(mode, index+1, round(time.perf_counter()-began, 2), flush=True)
                engine.close_segment()
            torch.cuda.synchronize()
            run = dict(timed_wall_seconds=time.perf_counter()-began, timings=profiler.result(),
                       counts=counts, cache=engine.cache.memory_stats(),
                       prefix_working_bytes=engine.encoding_prefix.working_bytes, timeline=timeline)
            report['runs'][mode] = run
            (output/'report.json').write_text(json.dumps(report, indent=2))
            # The following diagnostic does not advance canonical video memory.
            # It isolates the warm materialization and LLM step after the stream.
            engine.profiler = None
            engine.cache.profiler = None
            feature, _ = engine._frame_features(images[-1])
            def probe():
                with record_function('encoding_prefix'):
                    contexts = engine.encoding_prefix.contexts(engine.cache,
                        capture_scores=compressed, capture_frame_query=compressed and engine.scoring_strategy=='frame_mean_query')
                with record_function('llm_prefill'):
                    engine.decoder(inputs_embeds=feature, past_key_values=contexts, use_cache=True, return_dict=True)
                return contexts
            ctx = probe()
            prefix = ctx[0].past
            run['probe_layer0_lengths'] = ([h.key.shape[-2] for h in prefix.heads] if hasattr(prefix,'heads') else [prefix.key.shape[-2]])
            del ctx
            for _ in range(2):
                probe()
            torch.cuda.synchronize()
            durations = []
            for _ in range(5):
                t = time.perf_counter(); probe(); torch.cuda.synchronize()
                durations.append(time.perf_counter()-t)
            run['warm_probe_wall_seconds'] = durations
            if not args.skip_trace:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as trace:
                    for _ in range(3):
                        probe()
                        torch.cuda.synchronize()
                trace.export_chrome_trace(str(output/f'{mode}-trace.json'))
                run['probe_ops'] = aggregate(trace)
                # Attribute aten CPU dispatch time to the outer labelled phase.
                phases = {}
                for event in trace.events():
                    if not event.name.startswith('aten::'):
                        continue
                    parent = event.cpu_parent
                    while parent is not None and parent.name not in ('encoding_prefix','llm_prefill'):
                        parent = parent.cpu_parent
                    if parent is not None:
                        key = parent.name + '/' + event.name
                        row = phases.setdefault(key, dict(calls=0, cpu_self_ms=0., cuda_self_ms=0.))
                        row['calls'] += 1
                        row['cpu_self_ms'] += event.self_cpu_time_total/1000
                        row['cuda_self_ms'] += event.self_device_time_total/1000
                run['probe_phase_ops'] = phases
            if args.position_ablation:
                # Build a local variant for causal attribution only. Production
                # files and compression policy remain untouched. This encoding
                # branch reads already rotated keys and never reads positions.
                from model.suprkv import encoding
                source_code = textwrap.dedent(inspect.getsource(encoding.EncodingPrefix.contexts))
                source_code = source_code.replace('ids.long() + frame.rope_start', 'ids')
                source_code = source_code.replace('torch.cat([sink_positions] + [p[2] for p in parts])', 'sink_positions')
                scope = dict(vars(encoding))
                exec(compile(source_code, '<diagnostic: omit unused positions>', 'exec'), scope)
                variant = scope['contexts'].__get__(engine.encoding_prefix)
                reference_contexts = engine.encoding_prefix.contexts(engine.cache,
                    capture_scores=compressed, capture_frame_query=compressed)
                reference = engine.decoder(inputs_embeds=feature, past_key_values=reference_contexts,
                    use_cache=True, return_dict=True).last_hidden_state.clone()
                del reference_contexts
                with patch.object(engine.encoding_prefix, 'contexts', variant):
                    ablated_contexts = variant(engine.cache, capture_scores=compressed, capture_frame_query=compressed)
                    actual = engine.decoder(inputs_embeds=feature, past_key_values=ablated_contexts,
                        use_cache=True, return_dict=True).last_hidden_state
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                    del ablated_contexts, reference, actual
                    for _ in range(2):
                        probe()
                    torch.cuda.synchronize()
                    ablated = []
                    for _ in range(5):
                        t = time.perf_counter(); probe(); torch.cuda.synchronize()
                        ablated.append(time.perf_counter()-t)
                run['diagnostic_position_ablation'] = dict(wall_seconds=ablated, output_exact_match=True,
                    scope='warm final-prefix preparation and LLM prefill; not full-video speedup')
            report['runs'][mode] = run
            (output/'report.json').write_text(json.dumps(report, indent=2))
            print(mode, 'complete', flush=True)
        del engine
        torch.cuda.empty_cache()
    print(str(output/'report.json'), flush=True)


if __name__ == '__main__':
    main()
