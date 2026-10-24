"""Observation correctness checks; uses tiny random models, no downloads."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from observation.data import RVSDataset, decode_frames, sample_indices
from observation.engine import Cancelled, ObservationEngine, greedy_token_ids, install_observer
from observation.metrics import build_views, frame_statistics, window_importance
from observation.server import ObservationService, validate_params
from observation.spatial import video_token_layout


class WindowAttentionTests(unittest.TestCase):
    def test_gqa_and_full_causal_softmax_match_dense_attention(self):
        torch.manual_seed(12)
        query = torch.randn(1, 6, 45, 8)
        key = torch.randn(1, 2, 45, 8)
        dense = query @ key.repeat_interleave(3, dim=1).transpose(-1, -2) / np.sqrt(8)
        mask = torch.ones(45, 45, dtype=torch.bool).tril()
        dense.masked_fill_(~mask, -torch.inf)
        expected = dense.softmax(-1)[0, :, -32:].mean((0, 1)).numpy()
        actual = window_importance(query, key)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        self.assertAlmostEqual(float(actual.sum()), 1, places=6)
        # A softmax over historical keys alone would incorrectly sum to 1 here.
        self.assertLess(float(actual[:-32].sum()), 1)

    @torch.inference_mode()
    def test_observer_matches_native_attention_and_preserves_forward(self):
        torch.manual_seed(27)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=48,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
        native = Qwen2Model(config).eval()
        native.set_attn_implementation('eager')
        observed = copy.deepcopy(native)
        observed.set_attn_implementation('sdpa')
        ids = torch.randint(0, 64, (1, 49))
        expected = native(ids, use_cache=False, output_attentions=True)
        scores = {}
        cleanup = install_observer(observed, lambda layer, score: scores.update({layer: score}))
        try:
            actual = observed(ids, use_cache=False)
        finally:
            cleanup()
        torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, atol=2e-6, rtol=2e-5)
        for layer, attention in enumerate(expected.attentions):
            np.testing.assert_allclose(scores[layer], attention[0, :, -32:].mean((0, 1)).numpy(), atol=1e-7, rtol=2e-5)
        self.assertEqual(observed.config._attn_implementation, 'sdpa')
        self.assertFalse(hasattr(observed.layers[0].self_attn, '_rvs_observer'))

    def test_frame_mass_handles_window_overlap_and_empty_frames(self):
        frames = [dict(index=i, token_start=4*i, token_end=4*i+4) for i in range(3)]
        stats = frame_statistics([.01, .01, .01, .01, .2, 0], frames)
        self.assertEqual(stats, [
            dict(index=0, mass=.04, scored_tokens=4, window_tokens=0),
            dict(index=1, mass=.2, scored_tokens=2, window_tokens=2),
            dict(index=2, mass=0., scored_tokens=0, window_tokens=4)])
        self.assertEqual(frame_statistics(np.zeros(4), frames[:1])[0]['mass'], 0)

    def test_layer_aggregation_preserves_mass(self):
        frames = [dict(index=0, token_start=0, token_end=2)]
        views = build_views([[.4, 0, .6], [0, .4, .6]], frames, 2)
        np.testing.assert_allclose(views[0]['scores'], [.2, .2])
        self.assertAlmostEqual(views[0]['video_mass'], .4)
        self.assertAlmostEqual(views[0]['video_mass'] + views[0]['text_mass'] + views[0]['window_mass'], 1)


class GenerationAndSpatialTests(unittest.TestCase):
    @torch.inference_mode()
    def test_answer_reuses_observed_cache_and_matches_native_greedy_generation(self):
        torch.manual_seed(73)
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, eos_token_id=63, pad_token_id=0)).eval()
        model.set_attn_implementation('sdpa')
        ids = torch.randint(1, 60, (1, 54))
        captured = []
        cleanup = install_observer(model.model, lambda layer, scores: captured.append(scores.copy()))
        try:
            observed = model.model(ids[:, :49], use_cache=True)
        finally:
            cleanup()
        self.assertEqual(len(captured), 2)
        question = model.model(ids[:, 49:], past_key_values=observed.past_key_values, use_cache=True)
        direct = model.model(ids, use_cache=False)
        torch.testing.assert_close(question.last_hidden_state[:, -1], direct.last_hidden_state[:, -1], atol=2e-6, rtol=2e-5)
        tokens, reason = greedy_token_ids(model.model, model.lm_head, question, 4, {63})
        expected = model.generate(ids, max_new_tokens=4, do_sample=False)[0, ids.shape[1]:].tolist()
        self.assertEqual(tokens, expected)
        self.assertEqual(len(captured), 2)  # No answer queries enter the video window statistics.
        self.assertIn(reason, {'eos', 'max_new_tokens'})
        prefill = model.model(ids, use_cache=True)
        stop = int(model.lm_head(prefill.last_hidden_state[:, -1:])[0, -1].argmax())
        self.assertEqual(greedy_token_ids(model.model, model.lm_head, prefill, 4, {stop}), ([stop], 'eos'))

    def test_engine_keeps_both_window_modes_and_answer_generation(self):
        torch.manual_seed(74)
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)).eval()
        model.set_attn_implementation('sdpa')

        class Tokenizer:
            eos_token_id = None

            def __call__(self, text, **kwargs):
                return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3, 4]]))

            def decode(self, ids, **kwargs):
                return 'answer'

        def pixels(**kwargs):
            return SimpleNamespace(pixel_values_videos=torch.zeros(1, 2, 3, 8, 8))

        engine = ObservationEngine('.', device='cpu', dtype='float32')
        engine.model = SimpleNamespace(
            model=SimpleNamespace(language_model=model.model,
                get_video_features=lambda **kwargs: SimpleNamespace(pooler_output=torch.randn(1, 40, 32))),
            config=SimpleNamespace(vision_config=None, vision_feature_layer=-1, vision_feature_select_strategy='full'),
            generation_config=SimpleNamespace(eos_token_id=None), lm_head=model.lm_head,
            device='cpu', dtype=torch.float32)
        engine.processor = SimpleNamespace(tokenizer=Tokenizer(), video_processor=pixels)
        for mode, window in [('video', 8), ('question', 32)]:
            with self.subTest(mode=mode), patch('observation.engine.video_token_layout', return_value={'cells': []}):
                stages = []
                result = engine.observe(
                    [None, None], [dict(index=0), dict(index=1)],
                    dict(conversations=[dict(question='What happens?')]),
                    dict(mode=mode, video_window_size=8, question_index=0, max_new_tokens=2, segment_enabled=False),
                    lambda stage, *_: stages.append(stage), lambda: None)
                self.assertEqual(result['sequence']['window_size'], window)
                self.assertEqual(len(result['views']), 3)
                self.assertEqual(result['answer']['generated_tokens'], 2)
                self.assertEqual(result['answer']['input_tokens'], 48)
                self.assertEqual(stages.count('attention'), 2)
                self.assertEqual(set(result), {'sequence', 'model', 'method', 'frames', 'spatial',
                    'answer', 'views', 'observation_elapsed_seconds', 'elapsed_seconds'})
                self.assertEqual(set(result['views'][0]['frames'][0]),
                                 {'index', 'mass', 'scored_tokens', 'window_tokens'})
                self.assertEqual(model.model.config._attn_implementation, 'sdpa')
                json.dumps(result, allow_nan=False)

    @torch.inference_mode()
    def test_generation_can_be_cancelled_at_token_boundary(self):
        def cancel():
            raise Cancelled('cancelled')
        output = SimpleNamespace(past_key_values=object())
        with self.assertRaises(Cancelled):
            greedy_token_ids(None, None, output, 128, set(), check_cancel=cancel)

    def test_geometry_matches_actual_bilinear_pooling_and_source_projection(self):
        processor = SimpleNamespace(do_resize=True, do_center_crop=False, size={'height': 384, 'width': 384})
        vision = SimpleNamespace(image_size=384, patch_size=14)
        layout = video_token_layout(processor, vision, (384, 384), 196)
        self.assertEqual((layout['rows'], layout['cols']), (14, 14))
        self.assertEqual(layout['covered_box'], [0, 0, 378 / 384, 378 / 384])
        torch.manual_seed(10)
        patch_features = torch.randn(1, 3, 27, 27, dtype=torch.float64)
        expected = torch.nn.functional.interpolate(patch_features, size=(14, 14), mode='bilinear')
        for cell in layout['cells']:
            actual = sum(patch_features[0, :, p['row'], p['col']] * p['weight'] for p in cell['patches'])
            torch.testing.assert_close(actual, expected[0, :, cell['row'], cell['col']])
            self.assertAlmostEqual(sum(p['weight'] for p in cell['patches']), 1)
        # A non-square source is stretched by preprocessing; x/y map independently.
        last = layout['cells'][-1]['box']
        self.assertAlmostEqual(last[2] * 441, 441 * 378 / 384)
        self.assertAlmostEqual(last[3] * 240, 240 * 378 / 384)
        processor.do_center_crop = True
        with self.assertRaises(ValueError):
            video_token_layout(processor, vision, (384, 384), 196)


class SamplingTests(unittest.TestCase):
    def test_optional_frame_count_stops_after_sampling_and_start_filter(self):
        for kind, total, source_fps in (('npy', 1000, 1), ('video', 10000, 29.97)):
            full, available = sample_indices(total, source_fps, .5, kind, 2.1)
            for count in (1, 3, 256, 100000):
                selected, actual_available = sample_indices(total, source_fps, .5, kind, 2.1, count)
                self.assertEqual(selected, full[:count])
                self.assertEqual(actual_available, available)
        for bad in (0, -1, True, 1.5, '3'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sample_indices(100, 1, .5, 'npy', max_frames=bad)

    def test_sampling_matches_rekv_without_a_frame_cap(self):
        indices, available = sample_indices(100, 1, .5, 'npy', 4)
        original = np.linspace(0, 99, 50, dtype=int)
        np.testing.assert_array_equal(indices, original[original >= 4])
        self.assertEqual(available, sum(original >= 4))
        indices, _ = sample_indices(1000, 29.97, .5, 'video', 2.1)
        self.assertEqual(indices, list(range(120, 1000, 60)))
        indices, _ = sample_indices(1000, 1, 1, 'npy')
        self.assertEqual(len(indices), 1000)
        with self.assertRaises(ValueError):
            sample_indices(2, 1, .1, 'npy')
        with self.assertRaises(ValueError):
            sample_indices(20, 1, .5, 'npy', 100)

    def test_decoded_images_and_metadata_are_actual_model_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.stack([np.full((12, 16, 3), i * 10, dtype=np.uint8) for i in range(12)])
            np.save(root / 'video.npy', pixels)
            selected, frames, sampling = decode_frames(root / 'video.npy',
                dict(sample_fps=.5, start_time=2), root / 'images')
            self.assertEqual([f['source_index'] for f in frames], [2, 4, 6, 8, 11])
            self.assertEqual(sampling['processed_frames'], 5)
            self.assertFalse(sampling['truncated'])
            for frame, actual in zip(frames, selected):
                np.testing.assert_array_equal(actual, pixels[frame['source_index']])
                np.testing.assert_array_equal(np.asarray(Image.open(root / 'images' / frame['image'])), actual)
            selected, frames, sampling = decode_frames(root / 'video.npy',
                dict(sample_fps=.5, start_time=2, max_frames=3), root / 'limited')
            self.assertEqual([f['source_index'] for f in frames], [2, 4, 6])
            self.assertEqual(sampling['processed_frames'], 3)
            self.assertEqual(sampling['available_sampled_frames'], 5)
            self.assertTrue(sampling['truncated'])

    def test_dataset_validation_and_movie_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'movie' / 'videos').mkdir(parents=True)
            (root / 'movie' / 'movienet_oe.json').write_text(json.dumps([
                dict(video_id='example', video_path='old/path/movie.npy', conversations=[dict(question='q')])]))
            np.save(root / 'movie' / 'videos' / 'movie.npy', np.zeros((2, 2, 2, 3), dtype=np.uint8))
            dataset = RVSDataset(root)
            self.assertEqual(dataset.video_path(dataset.get('movie:0')), root / 'movie/videos/movie.npy')
            params = validate_params(dict(sample_id='movie:0'), dataset)
            self.assertEqual(params['mode'], 'video')
            for bad in [dict(max_frames=0), dict(max_frames=1.5), dict(sample_fps=float('nan')),
                        dict(start_time=-1), dict(mode='other'), dict(question_index=2), dict(sample_id='../etc'),
                        dict(max_new_tokens=0), dict(max_new_tokens=513), dict(max_new_tokens=1.5)]:
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    validate_params(dict(params, **bad), dataset)

    def test_completed_job_is_reused_and_reported_as_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'ego').mkdir()
            (root / 'ego/ego4d_oe.json').write_text(json.dumps([
                dict(video_id='sample', video_path='sample.mp4', conversations=[dict(question='q')])]))
            args = SimpleNamespace(data_root=root, cache_dir=root / 'cache', dtype='float16')
            service = ObservationService(args)
            try:
                service.jobs['cached-job'] = dict(id='cached-job', status='complete', cached=False)
                with patch.object(service, 'cache_key', return_value='cached-job'):
                    response = service.submit(dict(sample_id='ego:0'))
                self.assertTrue(response['cached'])
                self.assertEqual(response['status'], 'complete')
                self.assertFalse(service.events)  # No new GPU job was scheduled.
            finally:
                service.close()


if __name__ == '__main__':
    unittest.main()
