"""Summary Q / segment K diagnostics: GQA, softmax scope, and averaging order."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2Model

from observation.engine import Cancelled
from observation.segments import (TokenStateStore, capture_token_states, capture_rotary_states, mean_segment_attention,
                                  segment_attention)


class SegmentAttentionTests(unittest.TestCase):
    def test_gqa_matches_dense_and_softmax_precedes_mean(self):
        rng = np.random.default_rng(8)
        q, k = rng.normal(size=(7, 24)), rng.normal(size=(11, 8))
        dense_q = torch.tensor(q).reshape(7, 6, 4).permute(1, 0, 2)
        dense_k = torch.tensor(k).reshape(11, 2, 4).permute(1, 0, 2).repeat_interleave(3, dim=0)
        logits = dense_q @ dense_k.transpose(-1, -2) / 2
        probabilities = logits.softmax(-1)
        for chunk in [1, 3, 32]:
            before, after = mean_segment_attention(q, k, 6, 2, chunk)
            np.testing.assert_allclose(before, logits.mean((0, 1)).numpy(), atol=1e-12)
            np.testing.assert_allclose(after, probabilities.mean((0, 1)).numpy(), atol=1e-12)
            self.assertAlmostEqual(after.sum(), 1., places=12)
        self.assertFalse(np.allclose(after, logits.mean((0, 1)).softmax(-1).numpy()))

    def test_stability_uniform_and_invalid_states(self):
        pre, post = mean_segment_attention(np.zeros((2, 8)), np.ones((5, 4)), 4, 2)
        np.testing.assert_array_equal(pre, np.zeros(5))
        np.testing.assert_allclose(post, np.full(5, .2))
        _, post = mean_segment_attention(np.full((2, 8), 1e4), np.array([[1e4]*4, [-1e4]*4]), 4, 2)
        np.testing.assert_array_equal(post, [1., 0.])
        for q, k, heads, kv in [(np.ones((2, 8)), np.ones((3, 4)), 3, 2),
                                 (np.ones((2, 8)), np.ones((3, 6)), 4, 2),
                                 (np.full((2, 8), np.nan), np.ones((3, 4)), 4, 2)]:
            with self.assertRaises(ValueError):
                mean_segment_attention(q, k, heads, kv)

    def test_summary_key_mask_is_applied_before_softmax(self):
        q = np.array([[1., 0., 2., 0.], [1., 1., 2., 1.]])
        visual = np.array([[1., 0.], [0., 1.], [-1., 0.]])
        # Without a mask, these summary Keys would take essentially all probability.
        combined = np.concatenate([visual, np.full((2, 2), 1e6)])
        mask = np.array([True, True, True, False, False])
        before, after = mean_segment_attention(q, combined, 2, 1, key_mask=mask)
        expected = mean_segment_attention(q, visual, 2, 1)
        np.testing.assert_allclose(before[:3], expected[0])
        np.testing.assert_allclose(after[:3], expected[1])
        self.assertTrue(np.isneginf(before[3:]).all())
        np.testing.assert_array_equal(after[3:], [0., 0.])
        self.assertAlmostEqual(after[:3].sum(), 1.)
        self.assertLess(mean_segment_attention(q, combined, 2, 1)[1][:3].sum(), 1e-8)
        for invalid in [np.zeros(5, dtype=bool), np.ones(4, dtype=bool), np.ones(5)]:
            with self.assertRaises(ValueError):
                mean_segment_attention(q, combined, 2, 1, key_mask=invalid)

    @torch.inference_mode()
    def test_post_rope_capture_matches_native_rotation_and_segment_only_attention(self):
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        torch.manual_seed(14)
        model = Qwen2Model(Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=24,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)).eval()
        ids = torch.tensor([[1, 2, 3, 4]])
        positions = torch.tensor([[5, 6, 7, 8]])
        model.set_attn_implementation('eager')
        native = model(ids, position_ids=positions, use_cache=False, output_attentions=True)
        model.set_attn_implementation('sdpa')
        captured = {}
        with capture_rotary_states(model, lambda layer, q, k: captured.update(q=q.clone(), k=k.clone())):
            actual = model(ids, position_ids=positions, use_cache=False)
        torch.testing.assert_close(actual.last_hidden_state, native.last_hidden_state, atol=1e-6, rtol=1e-5)
        hidden = model.layers[0].input_layernorm(model.embed_tokens(ids))
        attn = model.layers[0].self_attn
        q = attn.q_proj(hidden).reshape(1, 4, 4, 4).transpose(1, 2)
        k = attn.k_proj(hidden).reshape(1, 4, 2, 4).transpose(1, 2)
        cos, sin = model.rotary_emb(hidden, positions)
        expected_q, expected_k = apply_rotary_pos_emb(q, k, cos, sin)
        torch.testing.assert_close(captured['q'], expected_q)
        torch.testing.assert_close(captured['k'], expected_k)
        self.assertFalse(torch.allclose(q, captured['q']))
        # First two tokens act as visual keys, last two as summary queries/keys.
        query = captured['q'][0, :, 2:].transpose(0, 1).reshape(2, -1).numpy()
        key = captured['k'][0].transpose(0, 1).reshape(4, -1).numpy()
        _, masked = mean_segment_attention(query, key, 4, 2, key_mask=np.array([True, True, False, False]))
        conditional = native.attentions[0][0, :, 2:, :2]
        conditional = conditional / conditional.sum(-1, keepdim=True)
        np.testing.assert_allclose(masked[:2], conditional.mean((0, 1)).numpy(), atol=1e-7)
        np.testing.assert_array_equal(masked[2:], [0., 0.])
        def cancel(*_):
            raise Cancelled()
        with self.assertRaises(Cancelled), capture_rotary_states(model, cancel):
            model(ids)
        self.assertEqual(model.config._attn_implementation, 'sdpa')

    @torch.inference_mode()
    def test_captures_actual_queries_and_removes_hooks_on_cancel(self):
        torch.manual_seed(10)
        model = Qwen2Model(Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=24,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)).eval()
        ids = torch.tensor([[1, 2, 3]])
        native = model(ids, use_cache=False).last_hidden_state
        captured = {}
        with capture_token_states(model, lambda kind, layer, x: captured.update({kind: x.clone()}), include_queries=True):
            actual = model(ids, use_cache=False).last_hidden_state
        expected = model.layers[0].self_attn.q_proj(model.layers[0].input_layernorm(model.embed_tokens(ids)))
        torch.testing.assert_close(captured['query'], expected)
        torch.testing.assert_close(actual, native)
        self.assertEqual(captured['query'].shape[-1], 16)
        self.assertEqual(captured['key'].shape[-1], 8)
        def stop(kind, *_):
            if kind == 'query':
                raise Cancelled()
        with self.assertRaises(Cancelled), capture_token_states(model, stop, include_queries=True):
            model(ids)
        self.assertFalse(model.layers[0].self_attn.q_proj._forward_hooks)
        self.assertFalse(model.layers[0].self_attn.k_proj._forward_hooks)
        self.assertFalse(model.layers[0]._forward_hooks)

    def test_frame_slices_preserve_full_segment_normalization_and_layer_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = [dict(index=i, token_start=1+i*2, token_end=3+i*2) for i in range(3)]
            keys, queries = TokenStateStore(directory), TokenStateStore(directory, 'query')
            q = torch.tensor([[1., 0., -1., 0.], [0., 1., 0., 1.]])
            visual = torch.tensor([[[9., 9.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.], [100., 0.], [0., 100.]]])
            for layer, factor in enumerate([1., 2.]):
                keys.video(layer, visual, frames)
                keys.summary(layer, 0, torch.full((2, 2), 1e4))
                queries.summary(layer, 0, q * factor)
            result = dict(model=dict(layers=2, heads=2, kv_heads=1), frames=frames,
                sequence=dict(tokens_per_frame=2), segment_experiment=dict(segment_attention={}, segments=[
                    dict(start_frame=0, end_frame=2, frame_indices=[0, 1], summary=dict(tokens=[{}, {}])),
                    dict(start_frame=2, end_frame=3, frame_indices=[2], summary=dict(tokens=[]))]))
            all_frames = [segment_attention(directory, result, 0, 0, f, 'all') for f in [0, 1]]
            expected = [mean_segment_attention(q.numpy()*factor, visual[0, 1:5].numpy(), 2, 1) for factor in [1., 2.]]
            np.testing.assert_allclose(np.concatenate([r['pre_softmax'] for r in all_frames]), np.mean([e[0] for e in expected], 0))
            np.testing.assert_allclose(np.concatenate([r['post_softmax'] for r in all_frames]), np.mean([e[1] for e in expected], 0))
            self.assertAlmostEqual(sum(sum(r['post_softmax']) for r in all_frames), 1.)
            self.assertLess(sum(all_frames[0]['post_softmax']), 1.)
            self.assertEqual(all_frames[0]['segment_tokens'], 4)
            self.assertEqual(all_frames[0]['masked_summary_tokens'], 2)
            self.assertEqual(all_frames[0]['summary_mass'], 0.)
            self.assertEqual(all_frames[0]['frame_masses'], all_frames[1]['frame_masses'])
            self.assertEqual(all_frames[1]['frame_token_start'], 3)
            single = segment_attention(directory, result, 0, 1, 2, '0')
            self.assertAlmostEqual(sum(single['post_softmax']), 1.)
            empty = segment_attention(directory, result, 1, 0, 0, 'all')
            self.assertEqual(empty['pre_softmax'], [None, None])
            self.assertEqual(empty['post_softmax'], [None, None])
            self.assertIsNone(empty['segment_mass'])
            json.dumps(empty, allow_nan=False)
            with self.assertRaises(ValueError):
                segment_attention(directory, result, 0, 0, 0, 'all', 'post')
            result['segment_experiment']['segment_attention']['rope_modes'] = ['pre', 'post']
            rotated_keys, rotated_queries = TokenStateStore(directory, 'key_rope'), TokenStateStore(directory, 'query_rope')
            for index in range(2):
                rotated_keys.video(index, visual, frames)
                rotated_keys.summary(index, 0, torch.full((2, 2), -1e4))
                rotated_queries.summary(index, 0, torch.flip(q, dims=[1]) * (index + 1))
            both = segment_attention(directory, result, 0, 0, 0, 'all', 'both')
            self.assertEqual([v['rope'] for v in both['variants']], ['pre', 'post'])
            self.assertFalse(np.allclose(both['variants'][0]['pre_softmax'], both['variants'][1]['pre_softmax']))
            for variant in both['variants']:
                self.assertEqual(variant['summary_mass'], 0.)
                self.assertAlmostEqual(variant['segment_mass'], 1.)
            json.dumps(both, allow_nan=False)
            with self.assertRaises(ValueError):
                segment_attention(directory, result, 0, 0, 0, 'all', '../post')
            for summary, segment, frame, layer in [(0, 0, 2, 'all'), (2, 0, 0, '0'), (0, 2, 0, '0'), (0, 0, 0, '../1')]:
                with self.assertRaises(ValueError):
                    segment_attention(directory, result, summary, segment, frame, layer)
            del result['segment_experiment']['segment_attention']
            with self.assertRaises(ValueError):
                segment_attention(directory, result, 0, 0, 0, '0')


if __name__ == '__main__':
    unittest.main()
