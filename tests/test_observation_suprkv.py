import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import numpy as np
from transformers import Qwen2Config, Qwen2ForCausalLM

from observation.suprkv import compare, retrieval_statistics
from observation.engine import Cancelled, ObservationEngine
from observation.server import validate_params, ObservationService
from model.suprkv.cache import tensor_storage_bytes
from model.suprkv.profiling import StageProfiler


class CompressionParameterTests(unittest.TestCase):
    def test_observation_loads_hub_id_without_resolving_as_local_path(self):
        model_id = 'llava-hf/llava-onevision-qwen2-7b-ov-hf'
        engine = ObservationEngine(model_id, device='cpu', dtype='float32')
        with patch('transformers.LlavaOnevisionProcessor.from_pretrained') as processor_load, \
             patch('transformers.LlavaOnevisionForConditionalGeneration.from_pretrained') as model_load:
            model = model_load.return_value.eval.return_value
            model.config.text_config.model_type = 'qwen2'
            engine.load(lambda *args: None)
            engine.load(lambda *args: None)
            processor_load.assert_called_once_with(model_id)
            model_load.assert_called_once_with(
                model_id, dtype=torch.float32, device_map='cpu',
                attn_implementation='sdpa', low_cpu_mem_usage=True,
            )
            self.assertIs(engine.model, model)

    def test_dtype_validation_and_unlimited_frame_parameters(self):
        dataset = SimpleNamespace(get=lambda _: dict(conversations=[dict(question='q')]))
        for dtype in ('float16', 'bfloat16', 'float32'):
            params = validate_params(dict(sample_id='sample', mode='compression', dtype=dtype), dataset)
            self.assertEqual(params['dtype'], dtype)
            self.assertIsNone(params['max_frames'])
        self.assertEqual(validate_params(dict(sample_id='sample'), dataset, 'bfloat16')['dtype'], 'bfloat16')
        self.assertEqual(validate_params(dict(sample_id='sample', segment_min_frames=10000), dataset)['segment_min_frames'], 10000)
        for count in (1, 128, 10000):
            self.assertEqual(validate_params(dict(sample_id='sample', max_frames=count), dataset)['max_frames'], count)
        self.assertEqual(validate_params(dict(sample_id='sample', max_frames=None), dataset),
                         validate_params(dict(sample_id='sample'), dataset))
        for bad in (dict(dtype='int8'), dict(dtype=None), dict(dtype=True), dict(dtype={}),
                    dict(max_frames=0), dict(max_frames=-1), dict(max_frames=1.5), dict(max_frames=True), dict(max_frames='8')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_params(dict(sample_id='sample', **bad), dataset)

    def test_service_reuses_same_dtype_and_releases_before_switching(self):
        import weakref
        service = object.__new__(ObservationService)
        service.args = SimpleNamespace(model_path='unused', device='cpu')
        service.engine = None
        created = []
        class Engine:
            def __init__(self, model_path, device, dtype):
                if created:
                    self.previous_released = created[-1]() is None
                self.dtype_name = dtype
                created.append(weakref.ref(self))
        with patch('observation.engine.ObservationEngine', Engine):
            service.select_dtype('float16')
            original = weakref.ref(service.engine)
            service.select_dtype('float16')
            self.assertEqual(len(created), 1)
            self.assertIs(original(), service.engine)
            service.select_dtype('bfloat16')
            self.assertTrue(service.engine.previous_released)
            self.assertIsNone(original())
            service.select_dtype('float32')
            self.assertTrue(service.engine.previous_released)
            self.assertEqual(service.engine.dtype_name, 'float32')

    def test_multiple_questions_validation(self):
        dataset = SimpleNamespace(get=lambda _: dict(conversations=[dict(question='one'), dict(question='two')]))
        raw = dict(sample_id='sample', mode='compression', question_indices=[1, 0], custom_questions=['  Custom?  '])
        params = validate_params(raw, dataset)
        self.assertEqual(params['question_indices'], [1, 0])
        self.assertEqual(params['custom_questions'], ['Custom?'])
        custom_only = validate_params(dict(raw, question_indices=[]), dataset)
        self.assertEqual(custom_only['question_indices'], [])
        for bad in [dict(question_indices=[] , custom_questions=[]), dict(question_indices=[True]),
                    dict(question_indices=[2]), dict(question_indices=[0, 0]), dict(question_indices='0'),
                    dict(custom_questions=[' ']), dict(custom_questions=['q'*2001]), dict(custom_questions='question'),
                    dict(custom_questions=['q']*32), dict(mode='question')]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_params(dict(raw, **bad), dataset)

    def test_retrieval_counts_deduplicate_layers_and_segment_members(self):
        frames = [dict(timestamp=i) for i in range(5)]
        run = dict(answers=[dict(hit_frames=[0, 0, 1, 3]), dict(hit_frames=[1, 2]), dict(hit_frames=[])])
        stats = retrieval_statistics(run, frames, {0:0, 1:0, 2:1, 3:1, 4:2})
        self.assertEqual([row['hits'] for row in stats['frame_hits']], [1, 2, 1, 1, 0])
        self.assertEqual([row['hits'] for row in stats['segment_hits']], [2, 2, 0])
        self.assertEqual([a['hit_segments'] for a in run['answers']], [[0, 1], [0, 1], []])
        missing_segments = retrieval_statistics(run, frames, {})
        self.assertEqual(missing_segments['segment_hits'], [])
        self.assertTrue(all(row['segment_id'] is None for row in missing_segments['frame_hits']))

    def test_parameters_and_invalid_budget_inputs(self):
        dataset = SimpleNamespace(get=lambda _: dict(conversations=[dict(question='q')]))
        params = validate_params(dict(sample_id='sample', mode='compression'), dataset)
        self.assertEqual(params['compression_budget_tokens'], 1024)
        for bad in [dict(compression_budget_tokens=0), dict(compression_budget_tokens=True),
                    dict(compression_topk=1, compression_recent_frames=2),
                    dict(compression_max_segment_frames=2, segment_min_frames=3)]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_params(dict(params, **bad), dataset)
        normal = validate_params(dict(sample_id='sample'), dataset)
        self.assertNotIn('compression_budget_tokens', normal)

    def test_cache_key_covers_compression_config_and_quantizer_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'data/movie/videos').mkdir(parents=True)
            (root/'model').mkdir()
            (root/'model/config.json').write_text('{}')
            (root/'data/movie/movienet_oe.json').write_text(json.dumps([
                dict(video_id='sample', video_path='sample.npy', conversations=[dict(question='q')])]))
            np.save(root/'data/movie/videos/sample.npy', np.zeros((2, 2, 2, 3), dtype=np.uint8))
            service = ObservationService(SimpleNamespace(data_root=root/'data', cache_dir=root/'cache',
                model_path=root/'model', device='cpu', dtype='float16'))
            try:
                params = validate_params(dict(sample_id='movie:0', mode='compression'), service.dataset)
                key = service.cache_key(params)
                self.assertNotEqual(key, service.cache_key(dict(params, dtype='bfloat16')))
                self.assertNotEqual(key, service.cache_key(dict(params, dtype='float32')))
                self.assertNotEqual(key, service.cache_key(dict(params, max_frames=1)))
                self.assertNotEqual(key, service.cache_key(dict(params, compression_budget_tokens=700)))
                self.assertNotEqual(key, service.cache_key(dict(params, custom_questions=['another question'])))
                read = Path.read_bytes
                def changed(path):
                    value = read(path)
                    return value+b'\n# changed quantizer' if path.name=='quantization.py' else value
                with patch.object(Path, 'read_bytes', changed):
                    self.assertNotEqual(key, service.cache_key(params))
                # Hub IDs need no local config and must identify distinct cached results.
                service.args.model_path = 'llava-hf/llava-onevision-qwen2-7b-ov-hf'
                hub_key = service.cache_key(params)
                self.assertNotEqual(key, hub_key)
                self.assertEqual(hub_key, service.cache_key(params))
                service.args.model_path = 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf'
                self.assertNotEqual(hub_key, service.cache_key(params))
            finally:
                service.close()

    def test_profile_nested_stages_are_exclusive(self):
        profiler = StageProfiler('cpu')
        with patch('model.suprkv.profiling.time.perf_counter', side_effect=[0., 1., 3., 5.]):
            with profiler.measure('parent'):
                with profiler.measure('child'):
                    pass
        self.assertEqual(profiler.seconds, {'child': 2., 'parent': 3.})
        self.assertEqual(sum(profiler.seconds.values()), 5.)
        x = torch.zeros(100)
        self.assertEqual(tensor_storage_bytes([x, x[:2]]), 400)
        self.assertEqual(tensor_storage_bytes([x, x[:2]], 'cuda'), 0)


class CompressionModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        self.llm = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=2048)).half().eval()
        self.llm.set_attn_implementation('sdpa')
        features = torch.randn(4, 196, 32).half()
        patches = torch.randn(4, 9, 4).half()
        class Tokenizer:
            eos_token_id = 31
            def __call__(self, text, **kwargs):
                return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))
            def decode(self, ids, **kwargs):
                return ' '.join(str(int(i)) for i in ids)
        def pixels(videos, **kwargs):
            return SimpleNamespace(pixel_values_videos=torch.tensor(videos))
        def visual(pixel_values, **kwargs):
            index = int(pixel_values[0])
            return SimpleNamespace(pooler_output=features[index:index+1], hidden_states=(patches[index:index+1],))
        self.observer = ObservationEngine('.', 'cpu', 'float16')
        self.observer.model = SimpleNamespace(model=SimpleNamespace(language_model=self.llm.model, get_video_features=visual),
            config=SimpleNamespace(vision_feature_layer=-1, vision_feature_select_strategy='full'),
            lm_head=self.llm.lm_head, dtype=torch.float16, parameters=self.llm.parameters)
        self.observer.processor = SimpleNamespace(tokenizer=Tokenizer(), video_processor=pixels)
        self.frames = [dict(index=i, timestamp=float(i)) for i in range(4)]
        self.record = dict(conversations=[dict(question='What happened?', answer='Never used as input')])
        self.params = dict(compression_budget_tokens=650, compression_topk=4, compression_recent_frames=2,
            compression_max_segment_frames=2, segment_min_frames=1, segment_ema_alpha=.2,
            summary_max_new_tokens=2, max_new_tokens=3, question_index=0)

    @torch.inference_mode()
    def test_bfloat16_and_float32_comparison_use_selected_model_dtype(self):
        original_visual = self.observer.model.model.get_video_features
        for dtype in (torch.bfloat16, torch.float32):
            self.llm.to(dtype)
            self.observer.model.dtype = dtype
            self.observer.dtype_name = str(dtype).removeprefix('torch.')
            def visual(**kwargs):
                self.assertEqual(kwargs['pixel_values'].dtype, dtype)
                output = original_visual(**kwargs)
                return SimpleNamespace(pooler_output=output.pooler_output.to(dtype),
                                       hidden_states=tuple(x.to(dtype) for x in output.hidden_states))
            self.observer.model.model.get_video_features = visual
            result = compare(self.observer, list(range(4)), self.frames, self.record,
                dict(self.params, compression_budget_tokens=5000), lambda *_: None, lambda: None)['compression_experiment']
            self.assertEqual(result['status'], 'complete', result['runs'])
            self.assertEqual(result['method']['dtype'], self.observer.dtype_name)
            for run in result['runs'].values():
                self.assertEqual(run['cache']['cache_dtype'], self.observer.dtype_name)
                self.assertTrue(run['answers'])

    @torch.inference_mode()
    def test_multiple_questions_share_video_cache_and_report_retrieval_hits(self):
        record = dict(conversations=self.record['conversations']+[dict(question='Second question?')])
        params = dict(self.params, question_indices=[1, 0], custom_questions=['Custom question?'])
        model = self.observer.model.model
        with patch.object(model, 'get_video_features', wraps=model.get_video_features) as visual:
            result = compare(self.observer, list(range(4)), self.frames, record, params,
                             lambda *_: None, lambda: None)['compression_experiment']
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(visual.call_count, 8)  # Four frames once on each side, not per question.
        self.assertEqual([q['question_index'] for q in result['questions']], [1, 0, None])
        self.assertEqual([q['question'] for q in result['questions']], ['Second question?', 'What happened?', 'Custom question?'])
        for run in result['runs'].values():
            self.assertEqual([a['question_id'] for a in run['answers']], [1, 2, 3])
            self.assertEqual(run['answer'], run['answers'][0])
            stats = run['retrieval_stats']
            self.assertEqual(stats['question_count'], 3)
            self.assertEqual(len(stats['frame_hits']), 4)
            for row in stats['frame_hits']:
                self.assertEqual(row['hits'], sum(row['frame_id'] in a['hit_frames'] for a in run['answers']))
            for row in stats['segment_hits']:
                self.assertEqual(row['hits'], sum(bool(set(row['frame_ids']) & set(a['hit_frames'])) for a in run['answers']))
        before, after = result['runs']['before'], result['runs']['after']
        self.assertEqual([f['hits'] for f in before['retrieval_stats']['frame_hits']], [3]*4)
        self.assertEqual([f['segment_id'] for f in before['retrieval_stats']['frame_hits']],
                         [f['segment_id'] for f in after['retrieval_stats']['frame_hits']])
        for frame in after['timeline'][-1]['frame_states']['frames']:
            if frame['status'] == 'evicted':
                self.assertEqual(after['retrieval_stats']['frame_hits'][frame['frame_id']]['hits'], 0)
        json.dumps(result, allow_nan=False)
        self.assertFalse(hasattr(self.llm.model, '_old_forward'))

    def test_cancel_between_questions_restores_model(self):
        def progress(stage, message, fraction):
            if '回答问题 2/' in message:
                raise Cancelled('cancel between questions')
        with self.assertRaises(Cancelled):
            compare(self.observer, list(range(4)), self.frames, self.record,
                    dict(self.params, custom_questions=['Another?']), progress, lambda: None)
        self.assertFalse(hasattr(self.llm.model, '_old_forward'))

    @torch.inference_mode()
    def test_paired_comparison_reuses_one_model_and_restores_observation(self):
        decoder = self.llm.model
        native = decoder(torch.tensor([[1, 2, 3]]), use_cache=False).last_hidden_state.clone()
        with patch('model.suprkv.onevision.LlavaOnevisionForConditionalGeneration.from_pretrained',
                   side_effect=AssertionError('Must reuse the Observation model')):
            result = compare(self.observer, list(range(4)), self.frames, self.record, self.params,
                             lambda *_: None, lambda: None)['compression_experiment']
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['method']['model_instances'], 1)
        before, after = result['runs']['before'], result['runs']['after']
        self.assertEqual(before['cache']['counts']['compressions'], 0)
        self.assertEqual(before['cache']['counts']['scoring_failures'], 0)
        self.assertLessEqual(after['cache']['persistent_bytes'], after['cache']['budget_bytes'])
        self.assertGreater(after['cache']['counts']['compressions'], 0)
        self.assertGreater(result['memory']['persistent_kv']['saved_bytes'], 0)
        self.assertIsNone(result['memory']['cuda_peak']['saved_bytes'])
        self.assertEqual(len(before['timeline']), 5)
        self.assertEqual(len(after['timeline']), 5)
        for run in (before, after):
            self.assertGreaterEqual(run['reporting_seconds'], 0)
            for point in run['timeline']:
                snapshot = point['frame_states']
                self.assertEqual(len(snapshot['frames']), point['frame_index']+1)
                self.assertEqual(snapshot['layer_count'], 2)
                for frame in snapshot['frames']:
                    if frame['status'] == 'evicted':
                        self.assertIsNone(frame['counts'])
                    else:
                        self.assertEqual(len(frame['counts']), 2)
                        self.assertTrue(all(0 <= sum(row) <= 196 for row in frame['counts']))
                        if frame['protected']:
                            self.assertEqual(frame['counts'], [[196, 0, 0, 0]]*2)
        baseline = before['timeline'][-1]['frame_states']
        self.assertTrue(all(frame['counts'] == [[196, 0, 0, 0]]*2 for frame in baseline['frames']))

        self.assertIn('quantization', after['timings'])
        self.assertNotIn('summary_generation', before['timings'])
        self.assertEqual(before['answer']['hit_frames'], [0, 1, 2, 3])
        self.assertFalse(hasattr(decoder, '_old_forward'))
        self.assertFalse(hasattr(decoder, 'position_bias'))
        self.assertFalse(hasattr(decoder.layers[0].self_attn, '_streaming_n_local'))
        torch.testing.assert_close(decoder(torch.tensor([[1, 2, 3]]), use_cache=False).last_hidden_state, native)

    @torch.inference_mode()
    def test_live_snapshots_preserve_segments_and_match_final_timeline(self):
        captured = {'before': [], 'after': []}
        def publish(side, point):
            captured[side].append(json.loads(json.dumps(point)))
        result = compare(self.observer, list(range(4)), self.frames, self.record, self.params,
                         lambda *_: None, lambda: None, on_snapshot=publish)['compression_experiment']
        self.assertEqual(result['status'], 'complete')
        for side in ('before', 'after'):
            self.assertEqual(captured[side], result['runs'][side]['timeline'])
        self.assertTrue(all(frame['segment_id'] is None for p in captured['before'] for frame in p['frame_states']['frames']))
        known = {}
        for point in captured['after']:
            for frame in point['frame_states']['frames']:
                self.assertIsInstance(frame['segment_id'], int)
                if frame['frame_id'] in known:
                    self.assertEqual(frame['segment_id'], known[frame['frame_id']])
                known[frame['frame_id']] = frame['segment_id']
        self.assertGreaterEqual(len(set(known.values())), 2)
        self.assertEqual(len(captured['after'][0]['frame_states']['frames']), 1)

    def test_cancel_from_live_snapshot_restores_native_model(self):
        def publish(side, point):
            if side == 'after':
                raise Cancelled('cancel live update')
        with self.assertRaises(Cancelled):
            compare(self.observer, list(range(4)), self.frames, self.record, self.params,
                    lambda *_: None, lambda: None, on_snapshot=publish)
        self.assertFalse(hasattr(self.llm.model, '_old_forward'))

    @torch.inference_mode()
    def test_no_compression_has_equal_payload_and_answers(self):
        result = compare(self.observer, list(range(4)), self.frames, self.record,
                         dict(self.params, compression_budget_tokens=10000), lambda *_: None, lambda: None)['compression_experiment']
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['memory']['persistent_kv']['saved_bytes'], 0)
        self.assertEqual(result['runs']['before']['answer']['answer'], result['runs']['after']['answer']['answer'])

    def test_invalid_budget_is_reported_per_side_and_restores_model(self):
        result = compare(self.observer, list(range(4)), self.frames, self.record,
                         dict(self.params, compression_budget_tokens=1), lambda *_: None, lambda: None)['compression_experiment']
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['memory'], {})
        self.assertEqual(result['runs']['after']['status'], 'failed')
        self.assertIn('cannot hold', result['runs']['after']['error'])
        self.assertFalse(hasattr(self.llm.model, '_old_forward'))
        self.assertIsNone(result['stages'][0]['after_seconds'])
        json.dumps(result, allow_nan=False)

    def test_cancellation_restores_model_and_does_not_return_partial_success(self):
        def cancel_on_frame(stage, message, fraction):
            if stage == 'compression_after':
                raise Cancelled('test cancel')
        with self.assertRaises(Cancelled):
            compare(self.observer, list(range(4)), self.frames, self.record, self.params,
                    cancel_on_frame, lambda: None)
        self.assertFalse(hasattr(self.llm.model, '_old_forward'))
        self.assertFalse(hasattr(self.llm.model, 'position_bias'))


if __name__ == '__main__':
    unittest.main()
