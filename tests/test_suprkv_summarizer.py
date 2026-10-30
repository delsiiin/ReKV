"""Text-only bridge, independent visual features, fallback and lifecycle regression tests."""
from dataclasses import asdict
from types import SimpleNamespace
import gc
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch
from transformers import (Qwen2Config, LlavaOnevisionConfig,
                          LlavaOnevisionForConditionalGeneration, LlavaOnevisionVideoProcessor)

from model.suprkv.onevision import OneVisionEngine, native_decoder
from model.suprkv.runtime import StreamingConfig, StreamingRuntime
from model.suprkv.summarizer import OneVisionSummarizer, DEFAULT_SUMMARIZER, SummarizerLoadError
from observation.engine import Cancelled, ObservationEngine
from observation.segments import capture_queries
from observation.suprkv import compare, run_pass
from observation.server import validate_params
from video_qa import base
from video_qa.methods.suprkv import SuPrKVMethod, make_config


class Tokenizer:
    eos_token_id = 31

    def __init__(self, caption):
        self.caption = caption
        self.inputs = []

    def __call__(self, text, **kwargs):
        self.inputs.append((text, kwargs))
        return SimpleNamespace(input_ids=torch.tensor([[3, 4, 5]]))

    def decode(self, tokens, **kwargs):
        return self.caption


def tiny_model(size=56, hidden=32, caption='main caption'):
    text = Qwen2Config(vocab_size=32, hidden_size=hidden, intermediate_size=48,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=4096)
    config = LlavaOnevisionConfig(text_config=text,
        vision_config=dict(model_type='siglip_vision_model', hidden_size=16, intermediate_size=24,
                           num_hidden_layers=2, num_attention_heads=4, image_size=size, patch_size=2),
        vision_feature_layer=-1, vision_feature_select_strategy='full')
    model = LlavaOnevisionForConditionalGeneration(config).eval()
    processor = SimpleNamespace(tokenizer=Tokenizer(caption), video_processor=LlavaOnevisionVideoProcessor(
        size={'shortest_edge': size}, crop_size={'height': size, 'width': size}))
    return model, processor


class SummarizerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model, self.processor = tiny_model()
        small, processor = tiny_model(8, 16, 'external caption')
        self.small = OneVisionSummarizer(small, processor, DEFAULT_SUMMARIZER)
        self.config = StreamingConfig(device='cpu', group_size=8, budget_tokens=10000,
            recent_frames=1, topk=2, min_segment_frames=4, max_segment_frames=4,
            summary_max_new_tokens=2, max_new_tokens=2, summarizer_model_path=DEFAULT_SUMMARIZER)
        self.image = np.full((56, 56, 3), 128, dtype=np.uint8)

    def engine(self, **updates):
        return OneVisionEngine(dict(asdict(self.config), **updates), model=self.model,
                               processor=self.processor, summarizer=self.small)

    def test_frame_query_strategy_captures_native_means_and_skips_all_summary_work(self):
        with patch.object(OneVisionSummarizer, 'load', side_effect=AssertionError('no loading')):
            engine = self.engine(scoring_strategy='frame_mean_query', summarizer_model_path='/missing/model')
        self.assertIsNone(engine.summarizer)
        captured = {}
        with (patch.object(engine, '_summary_length', side_effect=AssertionError('no summary context')),
                patch.object(engine.main_summarizer, 'generate', side_effect=AssertionError('no summary'))):
            with capture_queries(engine.decoder, lambda layer, q: captured.update({layer: q.detach().clone()})):
                engine.append_frame(self.image, 0.)
            for layer, q in captured.items():
                expected = q[0].float().reshape(196, 4, 8).mean(0)
                torch.testing.assert_close(engine.cache._score_queries[0][layer], expected)
            for i in range(1, 4):
                engine.append_frame(self.image, float(i))
        self.assertEqual(engine.cache.segment_id, 1)  # max-frame cap still closes the segment
        self.assertEqual(engine.stats()['model_instances'], 1)
        self.assertFalse(engine.stats()['last_event']['summary_enabled'])
        self.assertEqual(engine.stats()['last_event']['scoring_strategy'], 'frame_mean_query')
        self.assertFalse(engine.features or engine.summarizer_features or engine.cache._score_queries)
        self.assertEqual(engine.summary_counts['summary_failures'], 0)
        engine.cache.audit_accounting()
        engine.append_frame(self.image, 4.)
        self.assertTrue(engine.cache._score_queries)
        engine.reset()
        self.assertFalse(engine.cache._score_queries)

    def test_frame_query_fatal_scoring_failure_releases_means(self):
        engine = self.engine(scoring_strategy='frame_mean_query')
        for failure in (torch.cuda.OutOfMemoryError('oom'), Cancelled()):
            engine.reset()
            engine.append_frame(self.image, 0.)
            with patch.object(engine.cache, '_classify_segment', side_effect=failure):
                with self.assertRaises(type(failure)):
                    engine.close_segment()
            self.assertFalse(engine.cache._score_queries)
            self.assertFalse(engine.frame_info)
            engine.cache.audit_accounting()

    def test_different_hidden_sizes_bridge_through_main_text_queries(self):
        engine = self.engine()
        engine.append_frame(self.image, 0.)
        self.assertEqual(engine.features[0].shape, (1, 196, 32))
        self.assertEqual(engine.summarizer_features[0].shape, (1, 4, 16))
        self.assertEqual(engine.stats()['model_instances'], 2)
        self.assertGreater(engine.stats()['feature_working_bytes'], engine.features[0].numel()*4)
        with patch.object(engine.cache, 'close_segment', wraps=engine.cache.close_segment) as commit, \
                patch.object(engine.main_summarizer, 'generate', side_effect=AssertionError('no fallback')):
            result = engine.close_segment()
            queries = commit.call_args.args[0]
        self.assertEqual(result['summary'], 'external caption')
        self.assertEqual(result['summary_source'], DEFAULT_SUMMARIZER)
        self.assertEqual(result['summary_retries'], 0)
        self.assertIn(('external caption', dict(add_special_tokens=False, return_tensors='pt')), self.processor.tokenizer.inputs)
        expected = [None]*2
        def capture(i, states):
            expected[i] = states[0, 3:].reshape(-1, 4, 8)
        with torch.inference_mode(), native_decoder(engine.decoder), capture_queries(engine.decoder, capture):
            engine.decoder(input_ids=torch.tensor([[3, 4, 5, 3, 4, 5]]), use_cache=False)
        for actual, wanted in zip(queries, expected):
            torch.testing.assert_close(actual, wanted)
        self.assertFalse(engine.features or engine.summarizer_features)
        self.assertEqual(result['last_event']['summary'], 'external caption')
        self.assertEqual(result['model_parameter_bytes'], engine._model_parameter_bytes+self.small.parameter_bytes)

    def test_empty_and_failed_summary_retry_main_once(self):
        engine = self.engine()
        for failure in ('', RuntimeError('generation failed')):
            engine.reset()
            engine.append_frame(self.image, 0.)
            kwargs = {'side_effect': failure} if isinstance(failure, Exception) else {'return_value': failure}
            with patch.object(self.small, 'generate', **kwargs), \
                    patch.object(engine.main_summarizer, 'generate', wraps=engine.main_summarizer.generate) as retry:
                stats = engine.close_segment()
            self.assertEqual(retry.call_count, 1)
            self.assertEqual(stats['summary'], 'main caption')
            self.assertTrue(stats['summary_retry_succeeded'])
            self.assertEqual(stats['summary_retries'], 1)
            self.assertEqual(stats['summarizer_failures'], 1)
            self.assertEqual(stats['summary_failures'], 0)
            self.assertTrue(stats['summarizer_error'])
            self.assertEqual(stats['summary_error'], '')

    def test_visual_failure_stops_segment_encoding_and_resets(self):
        engine = self.engine()
        with patch.object(self.small, 'encode_frame', side_effect=ValueError('bad visual')) as encode:
            engine.append_frame(self.image, 0.)
            engine.append_frame(self.image, 1.)
            self.assertEqual(encode.call_count, 1)
        self.assertFalse(engine.summarizer_features)
        with patch.object(self.small, 'generate', side_effect=AssertionError('incomplete visual segment')):
            self.assertEqual(engine.close_segment()['summary'], 'main caption')
        engine.append_frame(self.image, 2.)
        self.assertEqual(len(engine.summarizer_features), 1)
        self.assertEqual(engine.summarizer_error, '')

    def test_double_failure_preserves_high_and_reports_error(self):
        engine = self.engine()
        engine.append_frame(self.image, 0.)
        with patch.object(self.small, 'generate', return_value=''), \
                patch.object(engine.main_summarizer, 'generate', side_effect=RuntimeError('main failure')):
            stats = engine.close_segment()
        self.assertEqual(stats['summary_failures'], 1)
        self.assertFalse(stats['summary_retry_succeeded'])
        self.assertIn('main failure', stats['summary_error'])
        self.assertEqual(stats['counts']['scoring_failures'], 1)
        self.assertFalse(engine.cache._score_keys)
        for frame in engine.cache.frames.values():
            self.assertTrue(frame.score_error)
        self.assertFalse(engine.features or engine.summarizer_features)

    def test_fatal_exceptions_propagate_without_retry(self):
        engine = self.engine()
        for error in (torch.cuda.OutOfMemoryError('oom'), Cancelled()):
            for phase in ('encode_frame', 'generate'):
                engine.reset()
                if phase == 'generate':
                    engine.append_frame(self.image, 0.)
                with patch.object(self.small, phase, side_effect=error), \
                        patch.object(engine.main_summarizer, 'generate') as retry:
                    with self.assertRaises(type(error)):
                        engine.close_segment() if phase == 'generate' else engine.append_frame(self.image, 0.)
                    retry.assert_not_called()

    def test_query_failure_does_not_regenerate_summary(self):
        engine = self.engine()
        engine.append_frame(self.image, 0.)
        engine.tokenizer = lambda *a, **kw: (_ for _ in ()).throw(ValueError('query tokenization'))
        with patch.object(engine.main_summarizer, 'generate') as retry:
            stats = engine.close_segment()
            retry.assert_not_called()
        self.assertIn('query tokenization', stats['summary_error'])
        self.assertEqual(stats['summary_retries'], 0)
        self.assertEqual(stats['counts']['scoring_failures'], 1)

    def test_external_context_boundary_and_one_frame_fallback(self):
        self.small.context_limit = 16  # Prompt 6 + two frames * 4 + reserve 2.
        engine = self.engine()
        for i in range(3):
            engine.append_frame(self.image, i)
        self.assertEqual(engine.cache.segment_id, 1)
        self.assertEqual(len(engine.features), 1)
        self.assertEqual(engine.last_summary['summary_source'], DEFAULT_SUMMARIZER)
        engine.close_segment()
        self.small.context_limit = 11
        engine.append_frame(self.image, 3.)
        stats = engine.close_segment()
        self.assertTrue(stats['summary_retried'])
        self.assertIn('capacity', stats['summarizer_error'])

    def test_main_context_and_max_frames_close_before_overflow(self):
        engine = self.engine()
        engine.main_summarizer.context_limit = 400 # Prompt 6 + 2*196 + reserve 2.
        for i in range(3):
            engine.append_frame(self.image, i)
        self.assertEqual(engine.cache.segment_id, 1)
        self.assertEqual(len(engine.features), 1)
        engine.reset()
        engine.main_summarizer.context_limit = 4096
        for i in range(4):
            engine.append_frame(self.image, i)
        self.assertEqual(engine.cache.segment_id, 1)
        self.assertFalse(engine.features)
        self.assertEqual(engine.close_segment()['last_summary']['segment_id'], 0)

    def test_runtime_load_once_reset_and_qa_after_summary(self):
        with patch('model.suprkv.onevision.LlavaOnevisionProcessor.from_pretrained', return_value=self.processor) as proc, \
                patch('model.suprkv.onevision.LlavaOnevisionForConditionalGeneration.from_pretrained', return_value=self.model) as main, \
                patch.object(OneVisionSummarizer, 'load', return_value=self.small) as small:
            with StreamingRuntime(self.config) as runtime:
                for _ in range(2):
                    runtime.append_frame(self.image, 0.)
                    runtime.close_segment()
                    self.assertEqual(runtime.ask('what')['answer'], 'main caption')
                    runtime.reset()
                    self.assertEqual(runtime.memory_stats()['summary_retries'], 0)
                self.assertEqual(runtime.startup['model_instances'], 2)
            proc.assert_called_once()
            main.assert_called_once()
            small.assert_called_once_with(DEFAULT_SUMMARIZER, torch.device('cpu'), torch.float32)

    def test_loader_errors_are_startup_failures(self):
        with patch('model.suprkv.summarizer.LlavaOnevisionProcessor.from_pretrained', side_effect=OSError('missing checkpoint')):
            with self.assertRaisesRegex(SummarizerLoadError, 'missing checkpoint'):
                OneVisionSummarizer.load('missing', 'cpu', torch.float32)
        with patch('model.suprkv.summarizer.LlavaOnevisionProcessor.from_pretrained', return_value=self.processor), \
                patch('model.suprkv.summarizer.LlavaOnevisionForConditionalGeneration.from_pretrained', side_effect=torch.cuda.OutOfMemoryError('oom')):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                OneVisionSummarizer.load('missing', 'cpu', torch.float32)

    def test_scoring_oom_propagates(self):
        engine = self.engine()
        engine.append_frame(self.image, 0.)
        with patch.object(engine.cache, '_classify_segment', side_effect=torch.cuda.OutOfMemoryError('score oom')):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                engine.close_segment()
        self.assertFalse(engine.features or engine.summarizer_features)

    def test_default_and_baseline_never_load_external(self):
        with patch.object(OneVisionSummarizer, 'load', side_effect=AssertionError('must not load')):
            baseline = OneVisionEngine(dict(asdict(self.config), compression_enabled=False),
                                      model=self.model, processor=self.processor)
            self.assertEqual(baseline.stats()['model_instances'], 1)
            baseline.append_frame(self.image, 0.)
            baseline.close_segment()
        # Use a fresh main decoder, since patch_hf must not be applied twice.
        model, processor = tiny_model()
        with patch.object(OneVisionSummarizer, 'load', side_effect=AssertionError('must not load')):
            engine = OneVisionEngine(dict(asdict(self.config), summarizer_model_path=None), model=model, processor=processor)
            engine.append_frame(self.image, 0.)
            self.assertEqual(engine.close_segment()['summary_source'], self.config.model_path)
            self.assertEqual(engine.stats()['model_instances'], 1)

    def test_observation_side_counts_records_and_restore(self):
        observer = ObservationEngine(self.config.model_path, 'cpu', 'float32')
        observer.model, observer.processor = self.model, self.processor
        original = self.model.model.language_model.forward
        params = dict(compression_budget_tokens=10000, compression_topk=2, compression_recent_frames=1,
            compression_max_segment_frames=2, segment_min_frames=2, segment_ema_alpha=.2,
            summary_max_new_tokens=2, max_new_tokens=2, question_index=0,
            summarizer_model_path=DEFAULT_SUMMARIZER)
        frames = [dict(index=i, timestamp=float(i)) for i in range(3)]
        with patch.object(OneVisionSummarizer, 'load', return_value=self.small) as load:
            result = compare(observer, [self.image]*3, frames,
                dict(conversations=[dict(question='what')]), params, lambda *_: None, lambda: None)['compression_experiment']
            load.assert_called_once()
        self.assertEqual(result['status'], 'complete', result['runs'])
        before, after = result['runs']['before'], result['runs']['after']
        self.assertEqual(before['model_stats']['model_instances'], 1)
        self.assertEqual(after['model_stats']['model_instances'], 2)
        self.assertEqual(before['summarizer_load_seconds'], 0)
        self.assertGreaterEqual(after['summarizer_load_seconds'], 0)
        self.assertEqual(len(after['commits']), 2)
        self.assertTrue(all(c['summary_source'] == DEFAULT_SUMMARIZER for c in after['commits']))
        self.assertIn('summarizer_vision_encoding', after['timings'])
        self.assertEqual(self.model.model.language_model.forward, original)
        self.assertFalse(hasattr(self.model.model.language_model, '_old_forward'))
        with torch.inference_mode():
            self.model.model.language_model(input_ids=torch.tensor([[3, 4, 5]]))

    def test_observation_cancellation_releases_auxiliary_and_restores_main(self):
        references = []
        def load(*args):
            model, processor = tiny_model(8, 16)
            auxiliary = OneVisionSummarizer(model, processor, DEFAULT_SUMMARIZER)
            references.append(weakref.ref(auxiliary))
            return auxiliary
        def cancel_on_frame(*args):
            raise Cancelled()
        decoder = self.model.model.language_model
        original = decoder.forward
        with patch.object(OneVisionSummarizer, 'load', side_effect=load):
            with self.assertRaises(Cancelled):
                run_pass(self.model, self.processor, [self.image], [dict(index=0, timestamp=0.)],
                         [dict(question_id=1, question='what')], asdict(self.config), True,
                         cancel_on_frame, lambda: None)
        gc.collect()
        self.assertIsNone(references[0]())
        self.assertEqual(decoder.forward, original)
        self.assertFalse(hasattr(decoder, '_old_forward'))


class SummarizerConfigTests(unittest.TestCase):
    def test_cli_forwarding_and_results(self):
        method = SuPrKVMethod()
        parser = base.build_parser('suprkv')
        for flag in ('--summarizer-model-path', '--summarizer_model_path'):
            args = parser.parse_args(['--anno_path', 'unused.json', '--save_dir', 'unused', flag, DEFAULT_SUMMARIZER])
            config = make_config(args)
            self.assertEqual(config.summarizer_model_path, DEFAULT_SUMMARIZER)
            forwarded = parser.parse_args(['--anno_path', 'unused.json', '--save_dir', 'unused', *method.worker_arguments(args)])
            self.assertEqual(make_config(forwarded).summarizer_model_path, DEFAULT_SUMMARIZER)
            self.assertIn('/summarizer_llava-onevision-qwen2-0.5b-ov-hf-', method.result_dir(args, 'rvs_ego'))
        args.summarizer_model_path = '/a/model'
        first = method.result_dir(args, 'rvs_ego')
        args.summarizer_model_path = '/b/model'
        self.assertNotEqual(first, method.result_dir(args, 'rvs_ego'))
        args.summarizer_model_path = None
        self.assertTrue(method.result_dir(args, 'rvs_ego').endswith('/budget_8192'))

    def test_observation_and_python_validation(self):
        positional = StreamingConfig('main-model', 'cpu')
        self.assertEqual(positional.device, 'cpu')
        self.assertIsNone(positional.summarizer_model_path)
        dataset = SimpleNamespace(get=lambda _: dict(conversations=[dict(question='q')]))
        for path in (None, DEFAULT_SUMMARIZER, ' /local/model '):
            params = validate_params(dict(sample_id='s', mode='compression', summarizer_model_path=path), dataset)
            expected = path.strip() if path else None
            self.assertEqual(params['summarizer_model_path'], expected)
            self.assertEqual(StreamingConfig(summarizer_model_path=path).summarizer_model_path, expected)
        for path in ('', '  ', True, 5, [], {}):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    validate_params(dict(sample_id='s', mode='compression', summarizer_model_path=path), dataset)
                with self.assertRaises(ValueError):
                    StreamingConfig(summarizer_model_path=path)
        normal = validate_params(dict(sample_id='s', summarizer_model_path=DEFAULT_SUMMARIZER), dataset)
        self.assertNotIn('summarizer_model_path', normal)


if __name__ == '__main__':
    unittest.main()
