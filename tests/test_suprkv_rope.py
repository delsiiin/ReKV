"""Post-RoPE storage, independent pre-RoPE scoring, and sliding/retrieval rebasing."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
import weakref

import torch

from model.rekv.attention.rope import RotaryEmbeddingESM
from model.rekv.attention.dot_production_attention.torch_impl import TorchMultiStageDotProductionAttention
from model.suprkv.attention import EncodeLayer, RetrieveLayer, RotaryCache, streaming_attention_forward
from model.suprkv.cache import StreamingKVCache
from model.suprkv.quantization import PackedKV
from model.suprkv.rotary import rotate_at
from model.suprkv.segments import summary_scores


def reference_rotate(x, positions, rope):
    """Independent rotation matrix reference, including partial rotary dimensions."""
    half = rope.inv_freq.numel()
    angle = positions[..., None].float() * rope.distance_scale * rope.inv_freq
    a, b = x[..., :half].float(), x[..., half:2*half].float()
    return torch.cat((a*angle.cos()-b*angle.sin(),
                      a*angle.sin()+b*angle.cos(), x[..., 2*half:].float()), -1).to(x.dtype)


def legacy_attention(q, raw_k, value, rope, window, sink):
    """Previous raw-K path, used only as an uncompressed attention oracle."""
    start = max(0, raw_k.shape[-2]-q.shape[-2]-window)
    local_k = raw_k[:, :, start:]
    lq, lk = rope(q, local_k)
    attention = TorchMultiStageDotProductionAttention(lq.shape, q.dtype, q.device)
    attention.append(lq, lk, value[:, :, start:], sliding_window=window)
    if raw_k.shape[-2] > window:
        iq = rope.apply_rotary_pos_emb_one_angle(q, window)
        ik, iv = raw_k[:, :, :sink], value[:, :, :sink]
    else:
        iq, ik, iv = q, raw_k[:, :, :0], value[:, :, :0]
    attention.append(iq, ik, iv, sliding_window=(raw_k.shape[-2]-q.shape[-2], window),
                     complement_sliding_window=True, end=True)
    return attention.get_result()[0]


class PostRoPETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.rope = RotaryEmbeddingESM(8, base=10000, distance_scale=.7)

    def forward(self, q, k, v, past=None, window=4, sink=2, fattn=False):
        # Explicit projections allow independent Q/K/V and GQA head counts.
        flatten = lambda x: x.transpose(1, 2).flatten(-2)
        fn = streaming_attention_forward(window, sink, fattn=fattn)
        output, updated = fn(SimpleNamespace(), flatten(q), flatten(k), self.rope, True, past,
            lambda _: flatten(q), lambda _: flatten(k), lambda _: flatten(v), lambda x: x,
            q.shape[-1], q.shape[1], k.shape[1])
        return output.view(1, q.shape[-2], q.shape[1], q.shape[-1]).transpose(1, 2), updated

    def test_incremental_attention_matches_raw_reference_across_window_rollover(self):
        past = None
        raw_k, values = torch.empty(1, 2, 0, 8), torch.empty(1, 2, 0, 8)
        for length in (2, 3, 3, 1, 1, 3):
            q, k, v = torch.randn(1, 4, length, 8), torch.randn(1, 2, length, 8), torch.randn(1, 2, length, 8)
            raw_k, values = torch.cat((raw_k, k), -2), torch.cat((values, v), -2)
            expected = legacy_attention(q, raw_k, values, self.rope, 4, 2)
            actual, past = self.forward(q, k, v, past)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            self.assertIsInstance(past, RotaryCache)
            if raw_k.shape[-2] > 6:
                raw_k = torch.cat((raw_k[:, :, :2], raw_k[:, :, -4:]), -2)
                values = torch.cat((values[:, :, :2], values[:, :, -4:]), -2)
            torch.testing.assert_close(past.key, reference_rotate(raw_k, past.positions, self.rope),
                                       rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(past.value, values)

    def test_capture_uses_actual_rotated_key_and_separate_raw_scoring_copy(self):
        sink_raw = torch.randn(1, 2, 2, 8)
        sink_v = torch.randn_like(sink_raw)
        sink_positions = torch.arange(2)
        context = EncodeLayer(RotaryCache(reference_rotate(sink_raw, sink_positions, self.rope),
                                        sink_v, sink_positions))
        q, k, v = torch.randn(1, 4, 4, 8), torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)
        actual, _ = self.forward(q, k, v, context)
        expected = legacy_attention(q, torch.cat((sink_raw, k), -2), torch.cat((sink_v, v), -2), self.rope, 4, 2)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(context.score_key, k[0], rtol=0, atol=0)
        torch.testing.assert_close(context.captured[0], reference_rotate(k, torch.arange(2, 6), self.rope)[0])
        torch.testing.assert_close(context.captured[1], v[0], rtol=0, atol=0)
        self.assertEqual(context.position_start, 2)
        self.assertFalse(torch.allclose(context.score_key, context.captured[0]))
        context.capture_scores, context.score_key = False, None
        self.forward(q, k, v, context)
        self.assertIsNone(context.score_key)

    def make_partitioned(self):
        cache = StreamingKVCache([(torch.zeros(2, 2, 8), torch.zeros(2, 2, 8))],
                                 100000, 4, group_size=8)
        raw = torch.zeros(2, 4, 8, dtype=torch.float16)
        raw[:, :, 0] = torch.tensor([3., -2., 1., -4.])
        post = reference_rotate(raw, torch.arange(10, 14), self.rope)
        value = torch.randn_like(raw)
        cache.append_frame(0., [(post, value)], score_keys=[raw], rope_start=10)
        return cache, raw, post, value

    def test_partition_scores_raw_but_packs_post_rope_and_releases_raw(self):
        cache, raw, post, value = self.make_partitioned()
        query = torch.ones(2, 4, 8)
        self.assertFalse(torch.equal(summary_scores(query, raw) >= summary_scores(query, raw).mean(),
                                     summary_scores(query, post) >= summary_scores(query, post).mean()))
        transient = weakref.ref(cache._score_keys[0][0])
        snapshot = cache.acquire_snapshot(1)
        cache.audit_accounting()
        cache.close_segment([query])
        self.assertFalse(cache._score_keys)
        self.assertIsNone(transient())  # Even an open-frame snapshot must not pin raw K.
        layer = cache.frames[0].layers[0]
        self.assertEqual(layer.high_indices.tolist(), [0, 2])
        torch.testing.assert_close(layer.high_k, post[:, [0, 2]], rtol=0, atol=0)
        torch.testing.assert_close(layer.high_v, value[:, [0, 2]], rtol=0, atol=0)
        torch.testing.assert_close(layer.representative, post.float().mean(1).half())
        packed = layer.drop_low().shrink(8)
        expected = PackedKV.quantize(post[:, [0, 2]], 8, axis='channel').shrink()
        torch.testing.assert_close(packed.high_k.data, expected.data, rtol=0, atol=0)
        torch.testing.assert_close(packed.high_k.scale, expected.scale, rtol=0, atol=0)
        torch.testing.assert_close(snapshot.frames[0].layers[0].high_k, post, rtol=0, atol=0)
        cache.audit_accounting()

    def test_pruned_retrieval_rebases_from_original_positions_and_decode_continues(self):
        cache, raw, post, value = self.make_partitioned()
        cache.close_segment([torch.ones(2, 4, 8)])
        frame = cache.frames[0]
        cache.frames[0] = replace(frame, layers=(frame.layers[0].drop_low().shrink(8),))
        snapshot = cache.acquire_snapshot(1)
        q, k, v = torch.randn(1, 4, 2, 8).half(), torch.randn(1, 2, 2, 8).half(), torch.randn(1, 2, 2, 8).half()
        keys, values, _, positions = snapshot.retrieve_layer(0, q, 2, position_bias=self.rope, with_rope_positions=True)
        self.assertEqual(positions.tolist(), [0, 1, 10, 12])
        hits = set()
        context = RetrieveLayer(snapshot, 0, 2, hits)
        actual, past = self.forward(q, k, v, context, window=64)
        # Invert the stored rotation only in the reference, then apply the old path.
        raw_cached = reference_rotate(keys, -positions, self.rope)
        expected = legacy_attention(q, torch.cat((raw_cached, k), -2), torch.cat((values, v), -2), self.rope, 64, 2)
        torch.testing.assert_close(actual, expected, rtol=.004, atol=.003)
        self.assertEqual(hits, {0})
        torch.testing.assert_close(past.key, keys, rtol=0, atol=0)
        self.assertEqual(past.key.shape[-2], 4)  # Retrieval query isn't persisted.
        actual, decoded = self.forward(q[:, :, :1], k[:, :, :1], v[:, :, :1], past, window=64)
        expected = legacy_attention(q[:, :, :1], torch.cat((raw_cached, k[:, :, :1]), -2),
                                    torch.cat((values, v[:, :, :1]), -2), self.rope, 64, 2)
        torch.testing.assert_close(actual, expected, rtol=.004, atol=.003)
        self.assertEqual(decoded.positions.tolist(), [0, 1, 10, 12, 4])

    def test_scoring_failure_and_uncompressed_commit_release_raw_keys(self):
        for mode in ('missing_summary', 'invalid_query', 'uncompressed'):
            cache, _, post, _ = self.make_partitioned()
            transient = weakref.ref(cache._score_keys[0][0])
            if mode == 'uncompressed':
                cache.close_uncompressed_segment()
            else:
                cache.close_segment(None if mode == 'missing_summary' else [torch.ones(1, 3, 8)])
            self.assertIsNone(transient())
            self.assertFalse(cache._score_keys)
            torch.testing.assert_close(cache.frames[0].layers[0].high_k, post)
            cache.audit_accounting()
        cache, _, _, _ = self.make_partitioned()
        cache._score_keys.clear()
        cache.close_segment([torch.ones(2, 4, 8)])
        self.assertIn('Missing pre-RoPE', cache.frames[0].score_error)
        self.assertEqual(cache.frames[0].layers[0].high_indices.numel(), 4)

    def test_partial_rope_and_negative_position_deltas(self):
        rope = RotaryEmbeddingESM(4, distance_scale=2)
        x = torch.randn(1, 2, 3, 8)
        positions = torch.tensor([-7, 0, 11])
        torch.testing.assert_close(rotate_at(x, positions, rope), reference_rotate(x, positions, rope))
        torch.testing.assert_close(rotate_at(x, positions, rope)[..., 4:], x[..., 4:], rtol=0, atol=0)

    def test_retrieval_scores_post_rope_representatives_with_aligned_queries(self):
        cache = StreamingKVCache([(torch.zeros(2, 2, 8), torch.zeros(2, 2, 8))],
                                 100000, 4, recent_frames=1, group_size=8)
        for i, start in enumerate((0, 7, 19, 32)):
            raw = torch.randn(2, 4, 8).half()
            post = reference_rotate(raw, torch.arange(start, start+4), self.rope)
            cache.append_frame(i, [(post, raw)], rope_start=start)
        cache.close_uncompressed_segment()
        snapshot = cache.acquire_snapshot(1)
        query = torch.randn(1, 4, 3, 8).half()
        scores = []
        for frame in snapshot.frames[:-1]:
            q = reference_rotate(query.float(), torch.arange(3)+frame.rope_start+4, self.rope).mean(-2)[0]
            representative = frame.layers[0].representative.float().repeat_interleave(2, 0)
            scores.append(float((q*representative).sum()))
        chosen = sorted(sorted(range(3), key=lambda i: scores[i], reverse=True)[:2]+[3])
        _, _, frame_ids, positions = snapshot.retrieve_layer(0, query, 3, position_bias=self.rope,
                                                            with_rope_positions=True)
        self.assertEqual(list(frame_ids), chosen)
        expected_positions = [0, 1]
        for i in chosen:
            start = cache.frames[i].rope_start
            expected_positions.extend(range(start, start+4))
        self.assertEqual(positions.tolist(), expected_positions)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_attention_with_post_rope_keys_and_window_rollover(self):
        self.rope = RotaryEmbeddingESM(16, distance_scale=.7).cuda()
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            past = None
            raw_k = torch.empty(1, 2, 0, 16, device='cuda', dtype=dtype)
            values = torch.empty_like(raw_k)
            for length in (2, 3, 3, 1):
                q = torch.randn(1, 4, length, 16, device='cuda', dtype=dtype)
                k, v = torch.randn_like(q[:, :2]), torch.randn_like(q[:, :2])
                raw_k, values = torch.cat((raw_k, k), -2), torch.cat((values, v), -2)
                expected = legacy_attention(q, raw_k, values, self.rope, 4, 2)
                actual, past = self.forward(q, k, v, past, fattn=dtype != torch.float32)
                tolerance = .025 if dtype == torch.bfloat16 else .005
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                self.assertEqual(past.key.dtype, dtype)
                self.assertEqual(past.key.device.type, 'cuda')
                self.assertEqual(past.positions.device.type, 'cuda')
                if raw_k.shape[-2] > 6:
                    raw_k = torch.cat((raw_k[:, :, :2], raw_k[:, :, -4:]), -2)
                    values = torch.cat((values[:, :, :2], values[:, :, -4:]), -2)

    @torch.inference_mode()
    def test_onevision_append_stores_rotated_keys_across_recent_frame_rollover(self):
        from transformers import Qwen2Config, Qwen2Model
        from model.suprkv.onevision import OneVisionEngine
        from unittest.mock import patch
        for compressed in (True, False):
            decoder = Qwen2Model(Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=48,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)).half().eval()
            features = torch.randn(1, 196, 32).half()
            model = SimpleNamespace(dtype=torch.float16, parameters=decoder.parameters,
                model=SimpleNamespace(language_model=decoder,
                    get_video_features=lambda **kwargs: SimpleNamespace(pooler_output=features)),
                config=SimpleNamespace(vision_feature_layer=-1, vision_feature_select_strategy='default'))
            tokenizer = lambda *args, **kwargs: SimpleNamespace(input_ids=torch.tensor([[1, 2]]))
            processor = SimpleNamespace(tokenizer=tokenizer,
                video_processor=lambda **kwargs: SimpleNamespace(pixel_values_videos=torch.zeros(1)))
            config = dict(device='cpu', recent_frames=2, group_size=8, budget_tokens=100000,
                          ema_alpha=.2, min_segment_frames=1, max_segment_frames=100,
                          summary_max_new_tokens=2, compression_enabled=compressed)
            engine = OneVisionEngine(config, model=model, processor=processor)
            engine.segmenter = SimpleNamespace(push=lambda _: False, count=1, max_frames=100)
            observed = []
            hook = decoder.layers[0].self_attn.k_proj.register_forward_hook(
                lambda module, args, output: observed.append(output.detach().clone()))
            try:
                with patch('model.suprkv.onevision.selected_vision_features', return_value=torch.zeros(1, 4, 32)):
                    for i, expected_start in enumerate((2, 198, 392, 392)):
                        engine.append_frame(None, float(i))
                        frame = engine.cache.frames[i]
                        raw = observed[-1][0].reshape(196, 2, 8).transpose(0, 1)
                        expected = reference_rotate(raw, torch.arange(expected_start, expected_start+196),
                                                    decoder.position_bias)
                        self.assertEqual(frame.rope_start, expected_start)
                        torch.testing.assert_close(frame.layers[0].high_k, expected, rtol=0, atol=0)
                        if compressed:
                            torch.testing.assert_close(engine.cache._score_keys[i][0], raw, rtol=0, atol=0)
                        else:
                            self.assertFalse(engine.cache._score_keys)
                        engine.cache.audit_accounting()
                if compressed:
                    engine.cache.close_segment([torch.ones(2, 4, 8)]*2)
                    self.assertFalse(engine.cache._score_keys)
                engine.reset()
                self.assertFalse(engine.cache.frames)
                self.assertFalse(engine.cache._score_keys)
            finally:
                hook.remove()


if __name__ == '__main__':
    unittest.main()
