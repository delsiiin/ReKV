import unittest
from dataclasses import replace
import numpy as np
import torch

from model.suprkv.quantization import PackedKV, pack, unpack
from model.suprkv.segments import OnlineSegmenter, summary_scores
from model.suprkv.cache import StreamingKVCache, LayerFrame, footprint
from observation.segments import segment_by_ema, vision_frame_distances, mean_segment_attention


def make_cache(frames=6, layers=2, tokens=16, dim=128):
    torch.manual_seed(9)
    cache = StreamingKVCache([(torch.randn(2, 2, dim), torch.randn(2, 2, dim)) for _ in range(layers)],
                             100000, tokens, group_size=128)
    queries = [torch.ones(3, 4, dim) * (-1 if layer % 2 else 1) for layer in range(layers)]
    for i in range(frames):
        data = []
        for layer in range(layers):
            key = torch.randn(2, tokens, dim) + torch.linspace(-2, 2, tokens)[None, :, None]
            data.append((key, torch.randn_like(key)))
        cache.append_frame(float(i), data, score_keys=[k for k, _ in data])
    cache.close_segment(queries)
    return cache, queries


class QuantizationTests(unittest.TestCase):
    def test_selected_dtype_survives_storage_scoring_and_all_compression_levels(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            x = torch.randn(2, 16, 128).to(dtype)
            # BF16 values outside FP16's range must survive protected storage.
            if dtype == torch.bfloat16:
                x[:, 0] = 100000
            cache = StreamingKVCache([(x[:, :2], x[:, :2])], 100000, 16, dtype=dtype)
            cache.append_frame(0., [(x, x)], score_keys=[x])
            self.assertEqual(cache.sink[0][0].dtype, dtype)
            self.assertEqual(cache._score_keys[0][0].dtype, dtype)
            layer = cache.frames[0].layers[0]
            torch.testing.assert_close(layer.high_k, x, rtol=0, atol=0)
            for bits in (16, 8, 4, 2):
                self.assertEqual(layer.bits, bits)
                key, value, _ = layer.materialize()
                self.assertEqual(key.dtype, dtype)
                self.assertEqual(value.dtype, dtype)
                self.assertTrue(torch.isfinite(key).all())
                self.assertTrue(torch.isfinite(value).all())
                layer = layer.shrink(128)
            cache.close_segment([torch.ones(2, 4, 128)])
            self.assertEqual(cache.frame_states()['full_precision_dtype'], str(dtype).removeprefix('torch.'))
            cache.audit_accounting()

    def test_channel_groups_use_tokens_and_exclude_padding_from_ranges(self):
        devices = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])
        for device in devices:
            # Noncontiguous input, distinct channels/heads, and a short last group.
            x = (torch.arange(2*10*19, device=device).reshape(2, 10, 19).float()+100).transpose(1, 2)
            x[:, :, 2] = 42
            p = PackedKV.quantize(x, 8, axis='channel')
            self.assertEqual(p.scale.shape, (2, 10, 3))
            self.assertEqual(p.data.shape, (2, 10, 24))
            for group, start in enumerate(range(0, 19, 8)):
                part = x[:, start:start+8, :]
                lo, hi = part.amin(1), part.amax(1)
                torch.testing.assert_close(p.offset[..., group], -lo)
                torch.testing.assert_close(p.scale[..., group], ((hi-lo)/65535).clamp(1e-5, 1e5))
            initial_scale = p.scale.clone()
            multiplier = 1
            for bits, factor in ((16, 1), (8, 257), (4, 17), (2, 5)):
                multiplier *= factor
                self.assertEqual(p.axis, 'channel')
                self.assertEqual(p.bits, bits)
                self.assertEqual(p.shape, tuple(x.shape))
                self.assertEqual(p.data.numel()*p.data.element_size(), 2*10*24*bits//8)
                torch.testing.assert_close(p.scale, initial_scale*multiplier)
                decoded = p.dequantize(dtype=torch.float32)
                self.assertEqual(decoded.shape, x.shape)
                self.assertTrue(decoded.is_contiguous())
                torch.testing.assert_close(decoded[:, :, 2], x[:, :, 2], rtol=0, atol=0)
                p = p.shrink()

    def test_channel_layout_matches_transposed_reference_through_all_bit_widths(self):
        devices = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])
        for device in devices:
            for tokens, channels, group in ((1, 10, 128), (7, 128, 128), (9, 10, 8), (33, 17, 24)):
                x = torch.randn(2, tokens, channels, device=device).half()
                p = PackedKV.quantize(x, group, axis='channel')
                reference = PackedKV.quantize(x.transpose(1, 2), group, axis='token')
                for bits in (16, 8, 4, 2):
                    with self.subTest(device=device, tokens=tokens, channels=channels, bits=bits):
                        self.assertEqual(p.axis, 'channel')
                        torch.testing.assert_close(p.data, reference.data, rtol=0, atol=0)
                        torch.testing.assert_close(p.scale, reference.scale, rtol=0, atol=0)
                        torch.testing.assert_close(p.offset, reference.offset, rtol=0, atol=0)
                        for dtype in (torch.float16, torch.float32):
                            torch.testing.assert_close(p.dequantize(dtype=dtype),
                                reference.dequantize(dtype=dtype).transpose(1, 2), rtol=0, atol=0)
                    p, reference = p.shrink(), reference.shrink()
        with self.assertRaisesRegex(ValueError, 'axis'):
            PackedKV.quantize(torch.ones(1, 8, 8), axis='invalid')

    def test_key_channels_and_value_tokens_have_independent_ranges(self):
        x = torch.arange(2*16*16).reshape(2, 16, 16).float()
        changed = x.clone()
        changed[0, 0, 0] = 10000
        for axis in ('channel', 'token'):
            before = PackedKV.quantize(x, 8, axis=axis)
            after = PackedKV.quantize(changed, 8, axis=axis)
            expected_changed = torch.zeros_like(before.scale, dtype=torch.bool)
            expected_changed[0, 0, 0] = True
            self.assertTrue(torch.equal(before.scale != after.scale, expected_changed))
            if axis == 'channel':
                torch.testing.assert_close(before.dequantize()[:, :, 1:], after.dequantize()[:, :, 1:])
            else:
                torch.testing.assert_close(before.dequantize()[:, 1:, :], after.dequantize()[:, 1:, :])

    def test_real_packing_and_integer_shrinking(self):
        for bits in (16, 8, 4, 2):
            codes = torch.arange(65536).reshape(2, 32768) % 2**bits
            packed = pack(codes, bits)
            self.assertEqual(packed.dtype, torch.int16)
            self.assertEqual(packed.numel()*packed.element_size(), codes.numel()*bits//8)
            torch.testing.assert_close(unpack(packed, bits), codes.int())
        x = torch.randn(2, 7, 130, dtype=torch.float16)
        p = PackedKV.quantize(x)
        self.assertEqual(p.scale.dtype, torch.float32)
        for bits in (16, 8, 4, 2):
            self.assertEqual(p.bits, bits)
            self.assertEqual(p.dequantize().shape, x.shape)
            self.assertTrue(torch.isfinite(p.dequantize()).all())
            p = p.shrink()

    def test_matches_local_pmkvq_codes_parameters_and_dequantization(self):
        import ast
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]/'PM-KVQ'
        if not root.is_dir():
            self.skipTest('Local PM-KVQ source is unavailable')
        sys.path.insert(0, str(root))
        try:
            from pm_kvq.quantization.quantizer.quantizer import UntrainableQuantizer
            from pm_kvq.quantization.quantizer.pack_funcs import pack_funcs, unpack_funcs
        finally:
            sys.path.pop(0)
        # Load the actual pure shrinking function without its obsolete CacheConfig imports.
        source = root/'pm_kvq/quantization/methods/pm_kvq/progressive/progressive_quantized_cache.py'
        tree = ast.parse(source.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name=='bit_width_shrinking')
        method.decorator_list = []
        namespace = dict(torch=torch, pack_funcs=pack_funcs, unpack_funcs=unpack_funcs)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        devices = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])
        for device in devices:
            torch.manual_seed(8)
            x = torch.randn(2, 7, 256, device=device)
            x[:, 0] = 3.  # constant groups
            x[:, 1] *= 1e-4  # PM-KVQ's minimum-scale clamp matters
            x[:, 2, 0] = 10000.  # no outlier calibration or exclusion
            p = PackedKV.quantize(x)
            reference = UntrainableQuantizer(n_bits=16, granularity='per_group', symmetric=False,
                                            group_size=128, round_zeros=False, pack=True)
            data, scale, offset = reference.real_quant(x, return_params=True)
            for bits in (16, 8, 4, 2):
                with self.subTest(device=device, bits=bits):
                    torch.testing.assert_close(p.data, data, rtol=0, atol=0)
                    torch.testing.assert_close(p.scale, scale, rtol=0, atol=0)
                    torch.testing.assert_close(p.offset, offset, rtol=0, atol=0)
                    decoder = UntrainableQuantizer(n_bits=bits, granularity='per_group', symmetric=False,
                                                  group_size=128, round_zeros=False, pack=True)
                    torch.testing.assert_close(p.dequantize(dtype=torch.bfloat16),
                                               decoder.dequant(data, scale, offset), rtol=0, atol=0)
                    if bits > 2:
                        data, scale, offset = namespace['bit_width_shrinking'](data, scale, offset, bits)
                        p = p.shrink()

    def test_first_compression_uses_int16_not_direct_int8(self):
        x = torch.linspace(0., .25, 128).reshape(1, 1, 128)
        initial = PackedKV.quantize(x)
        self.assertEqual(initial.bits, 16)
        torch.testing.assert_close(initial.scale, torch.full((1, 1, 1), 1e-5))
        compressed = initial.shrink()
        self.assertEqual(compressed.bits, 8)
        torch.testing.assert_close(compressed.scale, initial.scale*257)
        direct_scale = ((x.amax(-1, keepdim=True)-x.amin(-1, keepdim=True))/255).clamp(1e-5, 1e5)
        direct_codes = ((x-x.amin(-1, keepdim=True))/direct_scale).round().int()
        self.assertFalse(torch.equal(unpack(compressed.data, 8), direct_codes))
        # Both start at INT16, but K groups tokens and V groups channels.
        record = LayerFrame.full(x, x)
        compressed = PackedKV.quantize(x.half()).shrink()
        compressed_key = PackedKV.quantize(x.half(), axis='channel').shrink()
        record = record.shrink(128)
        self.assertEqual(record.bits, 8)
        torch.testing.assert_close(record.high_k.data, compressed_key.data, rtol=0, atol=0)
        torch.testing.assert_close(record.high_v.data, compressed.data, rtol=0, atol=0)
        torch.testing.assert_close(record.high_k.scale, compressed_key.scale, rtol=0, atol=0)
        torch.testing.assert_close(record.high_v.scale, compressed.scale, rtol=0, atol=0)


    def test_group_boundaries_and_padding_do_not_change_ranges(self):
        torch.manual_seed(2)
        x = torch.rand(2, 7, 10)*4+10
        p = PackedKV.quantize(x, group_size=8)
        expected_chunks = []
        for group, start in enumerate(range(0, x.shape[-1], 8)):
            part = x[..., start:start+8]
            lo, hi = part.amin(-1), part.amax(-1)
            scale = ((hi-lo)/65535).clamp(1e-5, 1e5)
            torch.testing.assert_close(p.scale[..., group], scale)
            torch.testing.assert_close(p.offset[..., group], -lo)
            codes = ((part-lo[..., None])/scale[..., None]).round().clamp(0, 65535)
            expected_chunks.append(codes*scale[..., None]+lo[..., None])
        torch.testing.assert_close(p.dequantize(dtype=torch.float32), torch.cat(expected_chunks, -1))
        for bits in (16, 8, 4, 2):
            self.assertEqual(p.data.shape[-1], 16*bits//16)
            self.assertEqual(p.dequantize().shape, x.shape)
            p = p.shrink()
        with self.assertRaises(ValueError):
            PackedKV.quantize(x, group_size=4)

    def test_constant_groups_and_storage_accounting(self):
        p = PackedKV.quantize(torch.full((1, 3, 128), 2., dtype=torch.float16))
        for bits in (16, 8, 4, 2):
            torch.testing.assert_close(p.dequantize(), torch.full((1, 3, 128), 2., dtype=torch.float16))
            p = p.shrink()
        tensor = torch.zeros(100)
        self.assertGreaterEqual(footprint(tensor[:1]), 400)
        self.assertLess(footprint([tensor, tensor[:1]]), footprint([tensor.clone(), tensor.clone()]))


class SegmentTests(unittest.TestCase):
    def test_incremental_matches_observation_and_preserves_ema(self):
        torch.manual_seed(5)
        features = torch.randn(12, 9, 4)
        distance, _ = vision_frame_distances(features)
        expected = segment_by_ema(distance, [dict(index=i, timestamp=i) for i in range(12)])
        segmenter = OnlineSegmenter(max_frames=100)
        starts = [0]
        for i, frame in enumerate(features):
            if segmenter.push(frame):
                starts.append(i)
        self.assertEqual(starts, [x['start_frame'] for x in expected['segments']])
        ema = segmenter.ema
        segmenter.close()
        self.assertEqual(segmenter.ema, ema)
        self.assertEqual(segmenter.count, 0)
        constant = OnlineSegmenter(max_frames=3)
        ends = []
        for i in range(8):
            self.assertFalse(constant.push(torch.ones(4, 2)))
            if constant.count >= constant.max_frames:
                ends.append(i)
                constant.close()
        self.assertEqual(ends, [2, 5])

    def test_scores_match_observation_per_layer(self):
        rng = np.random.default_rng(20)
        query, key = rng.normal(size=(5, 24)), rng.normal(size=(7, 8))
        expected, _ = mean_segment_attention(query, key, 6, 2)
        actual = summary_scores(torch.tensor(query).reshape(5, 6, 4),
                                torch.tensor(key).reshape(7, 2, 4).transpose(0, 1))
        self.assertEqual(actual.dtype, torch.float32)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-6)
        zero = summary_scores(torch.zeros(2, 4, 8), torch.ones(2, 5, 8))
        self.assertTrue((zero >= zero.mean()).all())
        with self.assertRaises(ValueError):
            summary_scores(torch.zeros(0, 4, 8), torch.ones(2, 5, 8))


class CacheTests(unittest.TestCase):
    def test_frame_states_report_each_layer_bit_width_and_eviction_without_mutation(self):
        import json
        cache, _ = make_cache(frames=4)
        original = cache.frame_states()
        self.assertEqual(original['bit_order'], [16, 8, 4, 2])
        self.assertEqual(original['layer_count'], 2)
        self.assertEqual(original['tokens_per_frame'], 16)
        self.assertTrue(all(row == [16, 0, 0, 0] for frame in original['frames'] for row in frame['counts']))
        first = cache.frames[0]
        cache.frames[0] = replace(first, layers=(first.layers[0].shrink(128), first.layers[1].shrink(128).shrink(128)))
        second = cache.frames[1]
        cache.frames[1] = replace(second, layers=(second.layers[0].shrink(128).shrink(128).shrink(128), second.layers[1].drop_low()))
        states = cache.frame_states()
        self.assertEqual(states['frames'][0]['counts'][0], [first.layers[0].low_indices.numel(), first.layers[0].high_indices.numel(), 0, 0])
        self.assertEqual(states['frames'][0]['counts'][1], [first.layers[1].low_indices.numel(), 0, first.layers[1].high_indices.numel(), 0])
        self.assertEqual(states['frames'][1]['counts'][0][-1], second.layers[0].high_indices.numel())
        self.assertEqual(sum(states['frames'][1]['counts'][1]), second.layers[1].high_indices.numel())
        self.assertEqual([f['frame_id'] for f in states['frames'] if f['protected']], [2, 3])
        known_segments = {f.frame_id:f.segment_id for f in cache.frames.values()}
        del cache.frames[0]
        self.assertEqual(cache.frame_states(known_segments)['frames'][0]['segment_id'], first.segment_id)
        evicted = cache.frame_states()['frames'][0]
        self.assertEqual(evicted, dict(frame_id=0, segment_id=None, status='evicted', protected=False, counts=None))
        # Earlier reports remain independent scalar data after later policy changes.
        self.assertEqual(original['frames'][0]['counts'], [[16, 0, 0, 0]]*2)
        json.dumps(states, allow_nan=False)
        cache.append_frame(4., [(torch.ones(2, 16, 128), torch.ones(2, 16, 128))]*2)
        current = cache.frame_states()
        self.assertEqual(current['frames'][-1]['status'], 'open')
        self.assertEqual(current['frames'][-1]['counts'], [[16, 0, 0, 0]]*2)
        self.assertEqual(len(current['frames']), 5)  # No future frame entries.

    def test_quantize_high_once_per_event_and_preserve_protected(self):
        cache, queries = make_cache()
        original = dict(cache.frames)
        # Set the first budget below the size after all low regions are removed.
        for i in range(4):
            frame = cache.frames[i]
            cache.frames[i] = replace(frame, layers=tuple(x.drop_low() for x in frame.layers))
        first_budget = cache.persistent_bytes()-1000
        cache.frames = dict(original)
        for expected in (8, 4, 2):
            cache.budget = first_budget if expected == 8 else cache.persistent_bytes()-1000
            cache._enforce_budget()
            self.assertLessEqual(cache.persistent_bytes(), cache.budget)
            self.assertEqual(len(cache.frames), 6)
            for frame in cache.frames.values():
                for layer, before in zip(frame.layers, original[frame.frame_id].layers):
                    if frame.frame_id in cache.recent_ids:
                        self.assertIs(layer, before)
                        self.assertEqual(layer.bits, 16)
                    else:
                        self.assertEqual(layer.bits, expected)
                        self.assertEqual(layer.high_k.axis, 'channel')
                        self.assertEqual(layer.high_v.axis, 'token')
                        self.assertEqual(layer.low_indices.numel(), 0)
                        self.assertIsNone(layer.low_k)
                        self.assertIsNone(layer.low_v)
                        torch.testing.assert_close(layer.high_indices, before.high_indices)
                        torch.testing.assert_close(layer.representative, before.representative, rtol=0, atol=0)
                        key, value, indices = layer.materialize()
                        torch.testing.assert_close(key, layer.high_k.dequantize())
                        torch.testing.assert_close(value, layer.high_v.dequantize())
                        torch.testing.assert_close(indices, before.high_indices)
            self.assertEqual(cache.counts['dropped_low_frames'], 4)
        self.assertEqual(cache.memory_stats()['bits'], {'16': 4, '8': 0, '4': 0, '2': 8, '0': 0})
        self.assertIs(cache.frames[0].layers[0].shrink(128), cache.frames[0].layers[0])
        cache.budget = 10**9
        data = [(torch.randn(2, 16, 128), torch.randn(2, 16, 128)) for _ in range(2)]
        cache.append_frame(6., data, score_keys=[k for k, _ in data])
        cache.close_segment(queries)
        self.assertEqual(cache.frames[6].layers[0].bits, 16)

    def test_drop_all_eligible_low_regions_before_quantization(self):
        from unittest.mock import patch
        cache, _ = make_cache()
        original = dict(cache.frames)
        cache.budget = cache.persistent_bytes()-1000
        with patch.object(PackedKV, 'quantize', side_effect=AssertionError('Low deletion suffices')):
            cache._enforce_budget()
        self.assertEqual(cache.counts['dropped_low_frames'], 4)
        self.assertEqual(cache.last_event['quantized'], 0)
        self.assertEqual(len(cache.frames), 6)
        for i, frame in cache.frames.items():
            for layer, before in zip(frame.layers, original[i].layers):
                if i in cache.recent_ids:
                    self.assertIs(layer, before)
                else:
                    self.assertEqual(layer.low_indices.numel(), 0)
                    self.assertIsNone(layer.low_k)
                    self.assertIsNone(layer.low_v)
                    self.assertEqual(layer.bits, 16)
                    torch.testing.assert_close(layer.high_k, before.high_k, rtol=0, atol=0)
                    torch.testing.assert_close(layer.high_v, before.high_v, rtol=0, atol=0)
        self.assertLessEqual(cache.persistent_bytes(), cache.budget)

    def test_budget_noop_open_frames_and_recent_transition(self):
        cache, queries = make_cache()
        original = dict(cache.frames)
        cache.budget = cache.persistent_bytes()
        cache._enforce_budget()
        self.assertTrue(all(cache.frames[i] is frame for i, frame in original.items()))
        for i in range(6, 9):
            data = [(torch.randn(2, 16, 128), torch.randn(2, 16, 128))]*2
            cache.append_frame(i, data, score_keys=[k for k, _ in data])
        opened = {i: cache.frames[i] for i in range(6, 9)}
        cache.budget = cache.persistent_bytes()-1000
        cache._enforce_budget()
        self.assertTrue(all(cache.frames[i] is frame for i, frame in opened.items()))
        self.assertTrue(all(not layer.low_indices.numel() for i in range(6) for layer in cache.frames[i].layers))
        cache.budget = 10**9
        cache.close_segment(queries)
        self.assertTrue(cache.frames[6].layers[0].low_indices.numel())
        cache.budget = cache.persistent_bytes()-1000
        cache._enforce_budget()
        self.assertFalse(cache.frames[6].layers[0].low_indices.numel())
        self.assertEqual(cache.frames[7].layers[0].bits, 16)
        self.assertEqual(cache.frames[8].layers[0].bits, 16)

    def test_all_high_fallback_can_be_compressed_without_pruning(self):
        cache, _ = make_cache(frames=0)
        for i in range(6):
            key = torch.randn(2, 16, 128)
            cache.append_frame(i, [(key, key)]*2)
        cache.close_segment(None)
        original = dict(cache.frames)
        cache.budget = cache.persistent_bytes()-1000
        cache._enforce_budget()
        self.assertEqual(cache.counts['dropped_low_frames'], 0)
        self.assertEqual(cache.last_event['quantized'], 8)
        self.assertEqual(len(cache.frames), 6)
        for i, frame in cache.frames.items():
            self.assertEqual(frame.score_error, 'summary_empty')
            for layer in frame.layers:
                self.assertEqual(layer.high_indices.numel(), 16)
                self.assertEqual(layer.bits, 16 if i in cache.recent_ids else 8)
            if i in cache.recent_ids:
                self.assertIs(frame, original[i])
        self.assertLessEqual(cache.persistent_bytes(), cache.budget)

    def test_quantized_high_materialization_preserves_original_token_order(self):
        key = torch.tensor([3., -2., 1., -4.]).view(1, 4, 1).expand(2, 4, 128)
        value = key + 10
        layer = LayerFrame.full(key, value).classify(torch.ones(3, 4, 128), key)
        self.assertEqual(layer.high_indices.tolist(), [0, 2])
        compressed = layer.shrink(128)
        k, v, indices = compressed.materialize()
        self.assertEqual(indices.tolist(), [0, 1, 2, 3])
        expected_key = key.half().clone()
        expected_key[:, [0, 2]] = compressed.high_k.dequantize()
        torch.testing.assert_close(k, expected_key, rtol=0, atol=0)
        torch.testing.assert_close(v, value.half(), rtol=0, atol=0)
        pruned = compressed.drop_low()
        self.assertEqual(pruned.bits, 8)
        k, v, indices = pruned.materialize()
        self.assertEqual(indices.tolist(), [0, 2])
        torch.testing.assert_close(k, expected_key[:, [0, 2]], rtol=0, atol=0)
        torch.testing.assert_close(v, value[:, [0, 2]].half(), rtol=0, atol=0)
        self.assertIs(pruned.drop_low(), pruned)

    def test_lfu_then_insertion_order_and_snapshot_is_immutable(self):
        cache, _ = make_cache()
        snap = cache.acquire_snapshot(1, cutoff=5.)
        original = snap.frames[0].layers[0].materialize()[0].clone()
        cache.record_activation(1, [0, 0, 2])
        cache.record_activation(1, [0, 1])
        self.assertEqual(cache.frames[0].hits, 1)
        self.assertEqual(cache.frames[1].hits, 0)
        active = dict(cache.frames)
        for i in range(4):
            frame = cache.frames[i]
            cache.frames[i] = replace(frame, layers=tuple(x.drop_low().shrink(128) for x in frame.layers))
        all_low_deleted = cache.persistent_bytes()
        cache.frames = active
        cache.budget = all_low_deleted-1000
        cache._enforce_budget()
        self.assertNotIn(1, cache.frames)  # oldest zero-hit frame, not frame 0
        self.assertIn(0, cache.frames)
        torch.testing.assert_close(snap.frames[0].layers[0].materialize()[0], original, rtol=0, atol=0)
        self.assertGreater(cache.memory_stats()['cache_working_bytes'], 0)
        cache.release_snapshot(1)
        self.assertFalse(cache.snapshots)

    def test_layer_masks_ragged_retrieval_recent_and_open_frames(self):
        cache, _ = make_cache()
        frame = cache.frames[0]
        cache.frames[0] = replace(frame, layers=tuple(x.drop_low() for x in frame.layers))
        cache.append_frame(6., [(torch.randn(2, 16, 128), torch.randn(2, 16, 128)) for _ in range(2)])
        with self.assertRaises(ValueError):
            cache.acquire_snapshot(1, cutoff=5.)
        snap = cache.acquire_snapshot(1, cutoff=6.)
        for layer in range(2):
            k, v, ids, positions = snap.retrieve_layer(layer, torch.ones(1, 4, 3, 128), topk=7)
            self.assertEqual(k.shape, v.shape)
            self.assertEqual(len(set(ids)), 7)
            self.assertIn(6, ids)
            self.assertEqual(k.shape[-2], 2+sum(len(x[1]) for x in positions))
        self.assertFalse(torch.equal(frame.layers[0].high_indices, frame.layers[1].high_indices))
        _, _, ids, _ = snap.retrieve_layer(0, torch.ones(1, 4, 3, 128), topk=2)
        self.assertEqual(ids, (5, 6))

    def test_eviction_uses_retrieve_frequency_then_fifo_with_equal_timestamps(self):
        for hits, expected_order in [((3, 1, 1, 2, 0, 0), (1, 2)),
                                     ((0, 0, 0, 0, 0, 0), (0, 1))]:
            with self.subTest(hits=hits):
                cache, _ = make_cache()
                for question_id in range(max(hits)):
                    cache.acquire_snapshot(question_id)
                    selected = [i for i, count in enumerate(hits) if count > question_id]
                    cache.record_activation(question_id, selected + selected)
                    cache.release_snapshot(question_id)
                self.assertEqual(tuple(f.hits for f in cache.frames.values()), hits)
                # Equal timestamps are allowed; insertion IDs must break hit ties.
                for i, frame in list(cache.frames.items()):
                    layers = frame.layers if i in cache.recent_ids else tuple(x.drop_low().shrink(128).shrink(128).shrink(128) for x in frame.layers)
                    cache.frames[i] = replace(frame, timestamp=5., layers=layers)
                # Mapping iteration order must not determine the eviction order.
                cache.frames = dict(reversed(list(cache.frames.items())))
                for victim in expected_order:
                    before = set(cache.frames)
                    cache.budget = cache.persistent_bytes()-500
                    cache._enforce_budget()
                    self.assertEqual(before-set(cache.frames), {victim})
                    self.assertTrue({4, 5}.issubset(cache.frames))
                    self.assertLessEqual(cache.persistent_bytes(), cache.budget)

    def test_tied_hits_evict_oldest_and_exact_commit_bytes(self):
        cache, _ = make_cache()
        saved = dict(cache.frames)
        for i in range(4):
            frame = cache.frames[i]
            cache.frames[i] = replace(frame, layers=tuple(x.drop_low().shrink(128) for x in frame.layers))
        cache.budget = cache.persistent_bytes()-500
        cache.frames = saved
        cache._enforce_budget()
        self.assertNotIn(0, cache.frames)
        self.assertIn(1, cache.frames)
        self.assertLessEqual(cache.persistent_bytes(), cache.budget)
        cache.budget = 10**9
        cache.append_frame(6., [(torch.randn(2, 16, 128), torch.randn(2, 16, 128)) for _ in range(2)])
        result = cache.close_segment(None)
        self.assertEqual(result['last_event']['after_bytes'], result['persistent_bytes'])
        self.assertEqual(result['last_event']['score_error'], 'summary_empty')

    def test_empty_summary_and_sink_only_and_budget_rejection(self):
        cache, _ = make_cache(frames=0)
        snap = cache.acquire_snapshot(0)
        k, v, ids, _ = snap.retrieve_layer(0, torch.ones(1, 4, 3, 128), topk=2)
        self.assertEqual(k.shape[-2], 2)
        self.assertEqual(ids, ())
        cache.release_snapshot(0)
        cache.append_frame(0., [(torch.ones(2, 16, 128), torch.zeros(2, 16, 128)) for _ in range(2)])
        cache.close_segment(None)
        self.assertEqual(cache.frames[0].score_error, 'summary_empty')
        self.assertEqual(cache.frames[0].layers[0].high_indices.numel(), 16)
        with self.assertRaises(ValueError):
            StreamingKVCache(cache.sink, 1, 16)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for no-offload coverage')
class ResidentCudaTests(unittest.TestCase):
    def test_cache_quantization_snapshot_and_retrieval_stay_on_cuda(self):
        from unittest.mock import patch
        from dataclasses import fields, is_dataclass
        device = torch.device('cuda:0')
        sink = [(torch.randn(2, 2, 128, device=device), torch.randn(2, 2, 128, device=device))]*2
        original_to = torch.Tensor.to
        def no_offload(tensor, *args, **kwargs):
            target = kwargs.get('device', args[0] if args else None)
            if isinstance(target, torch.Tensor):
                target = target.device
            if isinstance(target, (str, torch.device)) and tensor.is_cuda and torch.device(target).type == 'cpu':
                raise AssertionError('CUDA tensor offloaded to CPU')
            return original_to(tensor, *args, **kwargs)
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('Tensor.cpu is forbidden')), \
                patch.object(torch.Tensor, 'to', no_offload):
            cache = StreamingKVCache(sink, 100000, 16)
            for i in range(6):
                key = torch.randn(2, 16, 128, device=device)
                cache.append_frame(i, [(key, key), (key, key)], score_keys=[key, key])
            cache.close_segment([torch.ones(3, 4, 128, device=device)]*2)
            snap = cache.acquire_snapshot(1)
            original = snap.frames[0].layers[0].materialize()[0].clone()
            cache.budget = cache.persistent_bytes()-1000
            cache._enforce_budget()
            self.assertEqual(cache.frames[0].layers[0].bits, 16)
            for bits in (8, 4, 2):
                cache.budget = cache.persistent_bytes()-1000
                cache._enforce_budget()
                self.assertEqual(cache.frames[0].layers[0].bits, bits)
            def assert_device(value):
                if isinstance(value, torch.Tensor):
                    self.assertEqual(value.device, device)
                elif is_dataclass(value):
                    for field in fields(value):
                        assert_device(getattr(value, field.name))
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        assert_device(item)
            for frame in cache.frames.values():
                assert_device(frame)
            assert_device(cache.sink)
            torch.testing.assert_close(snap.frames[0].layers[0].materialize()[0], original)
            new_snapshot = type(snap)(cache.version, 2, cache.sink, tuple(cache.frames.values()), cache.recent_ids)
            key, value, _, positions = new_snapshot.retrieve_layer(0, torch.ones(1, 4, 2, 128, device=device))
            assert_device((key, value, [index for _, index in positions]))
            cache.release_snapshot(1)
            self.assertEqual(cache.memory_stats()['cache_device'], 'cuda:0')
            self.assertFalse(cache.memory_stats()['offload_enabled'])
            self.assertEqual(cache.frame_states()['frames'][0]['counts'][0][3], cache.frames[0].layers[0].high_indices.numel())
            with self.assertRaisesRegex(ValueError, 'offload'):
                cache.frames[0].layers[0].materialize('cpu')


if __name__ == '__main__':
    unittest.main()
