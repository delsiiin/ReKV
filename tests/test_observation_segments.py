"""Numerical and integration checks for Key segmentation / isolated summaries."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from observation.engine import Cancelled, ObservationEngine
from observation.segments import (EPSILON, capture_keys, capture_hidden_states, capture_token_states, cosine_matrix,
                                  selected_vision_features, vision_frame_distances,
                                  segment_by_ema, TokenStateStore, token_similarity, segment_attention, frame_query_attention)
from observation.server import validate_params


class SegmentMathTests(unittest.TestCase):
    def test_raw_vision_distance_and_cross_batch_continuity(self):
        rng = np.random.default_rng(3)
        raw = rng.normal(size=(7, 9, 12)).astype(np.float32)
        scale = np.sqrt(np.mean(np.sum(raw[1:] ** 2, axis=-1), axis=1) + EPSILON)
        expected = np.linalg.norm(raw[1:] - raw[:-1], axis=-1).mean(1) / scale
        distances, _ = vision_frame_distances(torch.tensor(raw))
        np.testing.assert_allclose(distances, expected, rtol=1e-6)
        first, previous = vision_frame_distances(torch.tensor(raw[:4]))
        second, _ = vision_frame_distances(torch.tensor(raw[4:]), previous)
        np.testing.assert_array_equal(np.concatenate([first, second]), distances)
        swapped = torch.tensor([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]])
        d, _ = vision_frame_distances(swapped)
        self.assertGreater(d[0], 1.4)  # Frame means match, spatial tokens do not.
        self.assertEqual(vision_frame_distances(torch.zeros_like(swapped))[0].tolist(), [0.])
        self.assertEqual(vision_frame_distances(swapped[:1])[0].tolist(), [])
        with self.assertRaises(ValueError):
            vision_frame_distances(torch.zeros(1, 3, 2), swapped[0])

    def test_vision_selection_uses_raw_layers_not_projected_features(self):
        a, b = torch.randn(3, 9, 4), torch.randn(3, 9, 4)
        output = SimpleNamespace(hidden_states=(a, b), pooler_output=torch.randn(1, 12, 8))
        torch.testing.assert_close(selected_vision_features(output, -1, 'full'), b)
        torch.testing.assert_close(selected_vision_features(output, [0, 1], 'default'), torch.cat([a, b], -1)[:, 1:])

    def test_ema_uses_past_only_and_preserves_every_frame_once(self):
        frames = [dict(index=i, timestamp=i * 2.) for i in range(6)]
        result = segment_by_ema([1., 1., 3., .5, 4.], frames, alpha=.5)
        self.assertEqual(result['multiplier'], 1.)
        self.assertEqual([t['threshold'] for t in result['transitions']], [None, 1., 1., 2., 1.25])
        self.assertEqual([(s['start_frame'], s['end_frame']) for s in result['segments']], [(0, 3), (3, 5), (5, 6)])
        self.assertEqual([i for s in result['segments'] for i in s['frame_indices']], list(range(6)))
        guarded = segment_by_ema([1., 1., 3., .5, 4.], frames, min_frames=4)
        self.assertEqual(guarded['transitions'][2]['reason'], 'min_frames')
        one = segment_by_ema([], frames[:1])
        self.assertEqual(one['segments'][0]['frame_indices'], [0])
        self.assertEqual(one['transitions'], [])
        zeros = segment_by_ema([0., 0., 1.], frames[:4])
        self.assertEqual(zeros['transitions'][1]['threshold'], EPSILON)
        self.assertTrue(zeros['transitions'][2]['boundary'])

    def test_signed_token_cosines_frame_selection_and_layer_averaging(self):
        result = cosine_matrix([[1., 0.], [0., 0.]], [[1., 0.], [-1., 0.], [0., 1.]])
        self.assertEqual(result, [[1., -1., 0.], [None, None, None]])
        with tempfile.TemporaryDirectory() as directory:
            store = TokenStateStore(directory)
            frames = [dict(index=0, token_start=1, token_end=3), dict(index=1, token_start=3, token_end=5)]
            # Same pooled image vector for both frames, but opposite token order.
            visual = torch.tensor([[[9., 9.], [1., 0.], [0., 1.], [0., 1.], [1., 0.], [8., 8.]]])
            for layer in range(2):
                store.video(layer, visual, frames)
            store.summary(0, 0, torch.tensor([[1., 0.], [0., 1.]]))
            store.summary(1, 0, torch.tensor([[1., 0.], [0., -1.]]))
            metadata = dict(model=dict(layers=2), frames=frames, sequence=dict(tokens_per_frame=2),
                segment_experiment=dict(token_similarity={}, segments=[dict(summary=dict(tokens=[
                    dict(index=0, token_id=4, text='one'), dict(index=1, token_id=5, text='two')]))]))
            first = token_similarity(directory, metadata, 0, 0, '0')
            second = token_similarity(directory, metadata, 0, 1, '0')
            self.assertEqual(first['cosine'], [[1., 0.], [0., 1.]])
            self.assertEqual(second['cosine'], [[0., 1.], [1., 0.]])
            self.assertEqual(second['frame_token_start'], 3)
            average = token_similarity(directory, metadata, 0, 0, 'all')
            self.assertEqual(average['cosine'], [[1., 0.], [0., 0.]])
            # Hidden states have a different dimension and must use a separate store.
            hidden_store = TokenStateStore(directory, 'hidden')
            hidden_visual = torch.tensor([[[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
            for layer in range(2):
                hidden_store.video(layer, hidden_visual, frames)
                hidden_store.summary(layer, 0, torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
            with self.assertRaises(ValueError):  # Old results never silently fall back to Keys.
                token_similarity(directory, metadata, 0, 0, 'all', 'hidden')
            metadata['segment_experiment']['token_similarity']['representations'] = ['key', 'hidden']
            hidden = token_similarity(directory, metadata, 0, 0, 'all', 'hidden')
            self.assertEqual(hidden['cosine'], [[0., 1.], [1., 0.]])
            self.assertEqual(hidden['representation'], 'hidden')
            with self.assertRaises(ValueError):
                token_similarity(directory, metadata, 0, 0, 'all', '../key')
            store.summary(1, 0, torch.zeros(2, 2))
            self.assertEqual(token_similarity(directory, metadata, 0, 0, 'all')['cosine'], [[None, None], [None, None]])
            for summary, frame, layer in [(1, 0, '0'), (-1, 0, '0'), (0, 2, 'all'), (0, 0, '2'), (0, 0, '../0')]:
                with self.assertRaises(ValueError):
                    token_similarity(directory, metadata, summary, frame, layer)
            metadata['segment_experiment']['segments'][0]['summary']['tokens'] = []
            empty = token_similarity(directory, metadata, 0, 0, 'all')
            self.assertEqual(empty['cosine'], [])
            json.dumps(empty, allow_nan=False)

    def test_api_validation(self):
        dataset = SimpleNamespace(get=lambda _: dict(conversations=[{}]))
        params = validate_params(dict(sample_id='sample'), dataset)
        self.assertTrue(params['segment_enabled'])
        self.assertEqual(params['segment_threshold_multiplier'], 1.)
        self.assertEqual(validate_params(dict(params, segment_threshold_multiplier=1), dataset), params)
        for name, bad_values in {
            'segment_enabled': [1, 'yes', None], 'segment_ema_alpha': [0, -1, 1.1, float('nan'), True],
            'segment_threshold_multiplier': [0, .8, 1.5, 2, True, None, float('inf'), float('nan'), '1'],
            'segment_min_frames': [0, 1.5, True], 'summary_max_new_tokens': [0, 513, True],
        }.items():
            for value in bad_values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    validate_params(dict(params, **{name: value}), dataset)


class SegmentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        torch.manual_seed(121)
        self.model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=32, hidden_size=24, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)).eval()
        self.model.set_attn_implementation('sdpa')

    @torch.inference_mode()
    def test_key_hook_preserves_native_output_and_cleans_up_on_cancel(self):
        decoder = self.model.model
        ids = torch.tensor([[1, 2, 3, 4]])
        native = decoder(ids).last_hidden_state
        observed = []
        with capture_keys(decoder, lambda layer, keys: observed.append(keys.clone())):
            actual = decoder(ids).last_hidden_state
        torch.testing.assert_close(actual, native)
        self.assertEqual(len(observed), 2)
        expected = decoder.layers[0].self_attn.k_proj(decoder.layers[0].input_layernorm(decoder.embed_tokens(ids)))
        torch.testing.assert_close(observed[0], expected)  # raw projection, before RoPE

        def cancel():
            raise Cancelled()
        with self.assertRaises(Cancelled), capture_keys(decoder, lambda *_: None, cancel):
            decoder(ids)
        self.assertTrue(all(not layer.self_attn.k_proj._forward_hooks for layer in decoder.layers))

    @torch.inference_mode()
    def test_hidden_states_are_block_outputs_and_hooks_clean_up(self):
        decoder = self.model.model
        ids = torch.tensor([[1, 2, 3, 4]])
        native = decoder(ids, output_hidden_states=True, use_cache=False)
        baseline_hooks = [dict(layer._forward_hooks) for layer in decoder.layers]
        captured = {}
        with capture_token_states(decoder, lambda kind, layer, x: captured.update({(kind, layer): x.clone()})):
            actual = decoder(ids, use_cache=False)
        torch.testing.assert_close(actual.last_hidden_state, native.last_hidden_state)
        torch.testing.assert_close(captured['hidden', 0], native.hidden_states[1])
        torch.testing.assert_close(decoder.norm(captured['hidden', 1]), native.last_hidden_state)
        self.assertEqual(captured['key', 0].shape[-1], 12)
        self.assertEqual(captured['hidden', 0].shape[-1], 24)

        def stop(kind, layer, tensor):
            if kind == 'hidden':
                raise Cancelled()
        with self.assertRaises(Cancelled), capture_token_states(decoder, stop):
            decoder(ids)
        for layer, baseline in zip(decoder.layers, baseline_hooks):
            self.assertEqual(dict(layer._forward_hooks), baseline)
            self.assertFalse(layer.self_attn.k_proj._forward_hooks)

    def test_full_experiment_and_independent_text_encoding(self):
        texts, calls = [], []

        class Tokenizer:
            eos_token_id = None

            def __call__(self, text, **kwargs):
                texts.append(text)
                return SimpleNamespace(input_ids=torch.tensor([[4, 5]] if text == 'Visible action.' else [[1, 2, 3]]))

            def decode(self, ids, **kwargs):
                return 'Visible action.'

        features = torch.randn(1, 40, 24)
        vision = torch.ones(4, 25, 7) * torch.tensor([1., 1.01, 2., .1])[:, None, None]
        engine = ObservationEngine('.', device='cpu', dtype='float32')
        engine.model = SimpleNamespace(
            model=SimpleNamespace(language_model=self.model.model,
                get_video_features=lambda **kwargs: SimpleNamespace(pooler_output=features, hidden_states=(vision,))),
            config=SimpleNamespace(vision_config=None, vision_feature_layer=-1, vision_feature_select_strategy='full'),
            generation_config=SimpleNamespace(eos_token_id=None), lm_head=self.model.lm_head,
            device='cpu', dtype=torch.float32)
        engine.processor = SimpleNamespace(tokenizer=Tokenizer(), video_processor=lambda **kwargs:
            SimpleNamespace(pixel_values_videos=torch.zeros(1, 4, 3, 8, 8)))

        def inspect(module, args, kwargs):
            calls.append(dict(ids=kwargs.get('input_ids'), embeds=kwargs.get('inputs_embeds'),
                              cached=kwargs.get('past_key_values') is not None, use_cache=kwargs.get('use_cache')))

        handle = self.model.model.register_forward_pre_hook(inspect, with_kwargs=True)
        try:
            with patch('observation.engine.video_token_layout', return_value={'cells': []}):
                result = engine.observe([None] * 4, [dict(index=i, timestamp=i * 2.) for i in range(4)],
                    dict(conversations=[dict(question='secret question', answer='secret reference')]),
                    dict(mode='video', video_window_size=8, question_index=0, max_new_tokens=2,
                         segment_enabled=True, summary_max_new_tokens=2),
                    lambda *_: None, lambda: None, artifact_dir=self.directory.name)
        finally:
            handle.remove()
        experiment = result['segment_experiment']
        self.assertEqual(experiment['vision_tokens_per_frame'], 25)
        self.assertEqual(experiment['vision_feature_dim'], 7)
        expected_distance, _ = vision_frame_distances(vision)
        np.testing.assert_allclose([t['distance'] for t in experiment['transitions']], expected_distance)
        self.assertNotIn('layer_distances', experiment)
        self.assertEqual(len(experiment['segments']), 3)
        self.assertEqual([len(s['frame_indices']) for s in experiment['segments']], [2, 1, 1])
        # Fresh summary prefills contain exactly this segment's visual features.
        prefills = [c for c in calls if c['embeds'] is not None][1:]
        self.assertEqual([c['embeds'].shape[1] for c in prefills], [26, 16, 16])
        for call, segment in zip(prefills, experiment['segments']):
            self.assertFalse(call['cached'])
            torch.testing.assert_close(call['embeds'][:, 3:-3], features[:, segment['start_frame'] * 10:segment['end_frame'] * 10])
            self.assertNotIn('secret', segment['summary']['prompt'])
        # Heatmap text encoding is fresh, text-only, and contains no generation prompt.
        standalone = [c for c in calls if c['use_cache'] is False]
        self.assertEqual(len(standalone), 3)
        for call in standalone:
            self.assertFalse(call['cached'])
            self.assertIsNone(call['embeds'])
            self.assertEqual(call['ids'].tolist(), [[1, 2, 3, 4, 5]])
        self.assertFalse(experiment['pooling'])
        self.assertEqual([t['token_id'] for t in experiment['segments'][0]['summary']['tokens']], [4, 5])
        self.assertEqual(np.load(Path(self.directory.name) / 'token_keys/video-0.npy').shape, (4, 10, 12))
        self.assertEqual(np.load(Path(self.directory.name) / 'token_keys/summary-0-0.npy').shape, (2, 12))
        self.assertEqual(np.load(Path(self.directory.name) / 'token_hidden/video-0.npy').shape, (4, 10, 24))
        self.assertEqual(np.load(Path(self.directory.name) / 'token_hidden/summary-0-0.npy').shape, (2, 24))
        self.assertEqual(np.load(Path(self.directory.name) / 'token_queries/summary-0-0.npy').shape, (2, 24))
        self.assertEqual(np.load(Path(self.directory.name) / 'token_queries_rope/summary-0-0.npy').shape, (2, 24))
        self.assertEqual(np.load(Path(self.directory.name) / 'token_keys_rope/video-0.npy').shape, (4, 10, 12))
        self.assertEqual(np.load(Path(self.directory.name) / 'token_keys_rope/summary-0-0.npy').shape, (2, 12))
        self.assertFalse((Path(self.directory.name) / 'token_queries/video-0.npy').exists())
        means = np.load(Path(self.directory.name) / 'frame_query_means/video-0.npy')
        self.assertEqual(means.shape, (4, 24))
        self.assertEqual(means.dtype, np.float32)
        frame_attention = frame_query_attention(self.directory.name, result, 0, 0, 'all')
        self.assertEqual(frame_attention['query_count'], 2)
        self.assertAlmostEqual(frame_attention['segment_mass'], 1.)
        matrix = token_similarity(self.directory.name, result, 0, 3, 'all')
        self.assertEqual(np.asarray(matrix['cosine']).shape, (2, 10))
        self.assertEqual(matrix['text_tokens'][1]['token_id'], 5)
        hidden_matrix = token_similarity(self.directory.name, result, 0, 3, 'all', 'hidden')
        self.assertEqual(np.asarray(hidden_matrix['cosine']).shape, (2, 10))
        self.assertFalse(np.allclose(matrix['cosine'], hidden_matrix['cosine']))
        attention = segment_attention(self.directory.name, result, 0, 0, 0, 'all')
        self.assertEqual(len(attention['pre_softmax']), 10)
        self.assertEqual(attention['segment_tokens'], 20)
        self.assertAlmostEqual(attention['segment_mass'], 1.)
        both = segment_attention(self.directory.name, result, 0, 0, 0, 'all', 'both')
        self.assertEqual(len(both['variants']), 2)
        self.assertEqual(both['positions']['summary_token_start'], 3)
        for variant in both['variants']:
            self.assertAlmostEqual(variant['segment_mass'], 1.)
            self.assertEqual(variant['summary_mass'], 0.)
        self.assertNotIn('secret reference', '\n'.join(texts))
        json.dumps(result, allow_nan=False)
        self.assertTrue(all(not layer.self_attn.k_proj._forward_hooks for layer in self.model.model.layers))
        self.assertTrue(all(not layer._forward_hooks for layer in self.model.model.layers))
        self.assertTrue(all(not layer.self_attn.q_proj._forward_hooks for layer in self.model.model.layers))
        self.assertEqual(self.model.model.config._attn_implementation, 'sdpa')


if __name__ == '__main__':
    unittest.main()
