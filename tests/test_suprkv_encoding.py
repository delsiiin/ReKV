"""Budget-triggered incremental encoding and reuse of the compressed prefix."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from model.rekv.attention.rope import RotaryEmbeddingESM
from model.suprkv.attention import streaming_attention_forward
from model.suprkv.cache import StreamingKVCache
from model.suprkv.encoding import EncodingPrefix
from model.suprkv.onevision import OneVisionEngine
from model.suprkv.packed_attention import HeadwiseRotaryCache
from model.suprkv.quantization import PackedKV
from model.suprkv.runtime import StreamingConfig
from test_suprkv_rope import reference_rotate
from test_suprkv_summarizer import tiny_model


class TinyEngine(OneVisionEngine):
    frame_tokens = 4


class EncodingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(235)

    def cache(self, granularity='layer'):
        sink = [(torch.randn(2, 2, 8), torch.randn(2, 2, 8))]
        cache = StreamingKVCache(sink, 10000, 16, recent_frames=1, group_size=8,
                                 dtype=torch.float32, compression_granularity=granularity)
        for i in range(3):
            k, v = torch.randn(2, 16, 8), torch.randn(2, 16, 8)
            cache.append_frame(float(i), [(k, v)], score_keys=[k], rope_start=2+16*i)
        cache.close_segment([torch.randn(3, 4, 8)])
        return cache

    @torch.inference_mode()
    def test_full_prefix_attention_matches_explicit_causal_reference(self):
        for granularity in ('layer', 'head'):
            with self.subTest(granularity=granularity):
                cache = self.cache(granularity)
                # Distinct masks per head, with original positions retained.
                old = cache.frames[0]
                entry = old.layers[0].drop_low().shrink(8)
                cache.frames[0] = replace(old, layers=(entry,))
                context = EncodingPrefix().contexts(cache, capture_scores=True, capture_frame_query=True)[0]
                q, k, v = torch.randn(1, 4, 4, 8), torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)
                rope = RotaryEmbeddingESM(8)
                positions = torch.arange(50, 54)
                rq, rk = reference_rotate(q, positions, rope), reference_rotate(k, positions, rope)
                heads = context.past.heads if isinstance(context.past, HeadwiseRotaryCache) else None
                expected = []
                for h in range(2):
                    pk = heads[h].key if heads else context.past.key[:, h:h+1]
                    pv = heads[h].value if heads else context.past.value[:, h:h+1]
                    keys, values = torch.cat((pk, rk[:, h:h+1]), -2), torch.cat((pv, v[:, h:h+1]), -2)
                    mask = torch.arange(keys.shape[-2])[None] <= pk.shape[-2] + torch.arange(4)[:, None]
                    expected.append(F.scaled_dot_product_attention(rq[:, h*2:h*2+2],
                        keys.repeat_interleave(2, 1), values.repeat_interleave(2, 1), attn_mask=mask))
                flatten = lambda x: x.transpose(1, 2).flatten(-2)
                # A tiny legacy window must not truncate the full-prefix path.
                fn = streaming_attention_forward(4, 2, fattn=False)
                actual, updated = fn(SimpleNamespace(), flatten(q), flatten(k), rope, True, context,
                    lambda _: flatten(q), lambda _: flatten(k), lambda _: flatten(v), lambda x: x, 8, 4, 2)
                torch.testing.assert_close(actual, flatten(torch.cat(expected, 1)), rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(context.captured[0], rk[0])
                torch.testing.assert_close(context.score_key, k[0])
                self.assertIs(updated, context)
                self.assertEqual(context.position_start, 50)

    @torch.inference_mode()
    def test_dequantization_reuse_and_invalidation(self):
        for granularity in ('layer', 'head'):
            with self.subTest(granularity=granularity):
                cache = self.cache(granularity)
                frame = cache.frames[0]
                cache.frames[0] = replace(frame, layers=(frame.layers[0].drop_low().shrink(8),))
                prefix = EncodingPrefix()
                kwargs = dict(capture_scores=True, capture_frame_query=False)
                original = PackedKV.dequantize
                with patch.object(PackedKV, 'dequantize', autospec=True, side_effect=original) as decode:
                    prefix.contexts(cache, **kwargs)
                    count = decode.call_count
                    self.assertGreater(count, 0)
                    self.assertGreater(prefix.working_bytes, 0)
                    # Appending raw frames or updating hit counts does not decode old frames again.
                    k = torch.randn(2, 16, 8)
                    cache.append_frame(3., [(k, k)], score_keys=[k], rope_start=50)
                    frame = cache.frames[0]
                    cache.frames[0] = replace(frame, hits=frame.hits+1)
                    prefix.contexts(cache, **kwargs)
                    self.assertEqual(decode.call_count, count)
                    cache.frames[0] = replace(frame, layers=(frame.layers[0].shrink(8),))
                    prefix.prune(cache.frames)
                    self.assertNotIn((0, 0), prefix.entries)
                    prefix.contexts(cache, **kwargs)
                    self.assertGreater(decode.call_count, count)
                    del cache.frames[0]
                    prefix.prune(cache.frames)
                    self.assertNotIn((0, 0), prefix.entries)
                prefix.clear()
                self.assertEqual(prefix.working_bytes, 0)

    @torch.inference_mode()
    def test_budget_interrupts_pending_batch_and_subsequent_frames_use_survivors(self):
        for granularity in ('layer', 'head'):
            with self.subTest(granularity=granularity):
                model, processor = tiny_model(size=8)
                config = StreamingConfig(device='cpu', budget_tokens=400, group_size=8, recent_frames=1,
                    topk=2, scoring_strategy='frame_mean_query', scoring_scope='frame',
                    frame_score_batch_size=100, compression_granularity=granularity)
                engine = TinyEngine(asdict(config), model=model, processor=processor)
                engine._frame_features = lambda _: (torch.randn(1, 4, 32), None)
                seen_pressure, seen_context = [], []
                original_enforce = engine.cache._enforce_budget
                original_forward = engine.decoder.forward

                def enforce():
                    seen_pressure.append(engine.cache.persistent_bytes())
                    return original_enforce()

                def forward(*args, **kwargs):
                    contexts = kwargs['past_key_values']
                    frames = list(engine.cache.frames.values())
                    for layer, context in enumerate(contexts):
                        past = context.past
                        heads = past.heads if isinstance(past, HeadwiseRotaryCache) else None
                        for h in range(2):
                            expected = engine.sink_tokens + sum(
                                f.layers[layer].materialize_head(h)[0].shape[1] for f in frames)
                            actual = heads[h].key.shape[-2] if heads else past.key.shape[-2]
                            self.assertEqual(actual, expected)
                        self.assertEqual(context.next_position, engine.sink_tokens+engine.cache.next_id*4)
                    seen_context.append(len(frames))
                    return original_forward(*args, **kwargs)

                with patch.object(engine.cache, '_enforce_budget', side_effect=enforce), \
                        patch.object(engine.decoder, 'forward', side_effect=forward):
                    for i in range(32):
                        engine.append_frame(None, float(i))
                        self.assertLess(engine.cache.persistent_bytes(), engine.cache.budget)
                        engine.cache.audit_accounting()
                        if not seen_pressure:
                            self.assertEqual(len(engine.cache.frames), i+1)
                            self.assertTrue(all(x.bits == 16 for f in engine.cache.frames.values() for x in f.layers))
                self.assertGreater(len(seen_pressure), 1)
                self.assertTrue(all(value >= engine.cache.budget for value in seen_pressure))
                self.assertGreater(max(seen_context), config.recent_frames)
                self.assertGreater(engine.cache.counts['compressions'], 1)
                self.assertTrue(engine.encoding_prefix.entries)
                engine.reset()
                self.assertFalse(engine.encoding_prefix.entries)
                self.assertEqual(engine.stats()['encoding_prefix_working_bytes'], 0)

    def test_exact_byte_threshold_triggers_compression(self):
        cache = self.cache()
        cache.budget = cache.persistent_bytes()
        cache._enforce_budget()
        self.assertEqual(cache.counts['compressions'], 1)
        self.assertLess(cache.persistent_bytes(), cache.budget)


if __name__ == '__main__':
    unittest.main()
