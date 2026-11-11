"""CPU archives preserve compressed KV and remain searchable after eviction."""
from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.suprkv.cache import LayerFrame, StreamingKVCache, tensor_storage_bytes
from model.suprkv.runtime import StreamingConfig, StreamingRuntime
from observation.server import validate_params
from video_qa import base
from video_qa.methods.suprkv import SuPrKVMethod, make_config


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from tensors(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)


def make_cache(granularity='layer', scope='segment', enabled=True, device='cpu'):
    torch.manual_seed(219)
    sink = torch.zeros(2, 2, 32, device=device)
    cache = StreamingKVCache([(sink, sink)] * 2, 100000, 16, recent_frames=1, group_size=8,
                             dtype=torch.float32, compression_granularity=granularity,
                             scoring_scope=scope, offload_enabled=enabled)
    for i in range(5):
        keys = [torch.randn(2, 16, 32, device=device) for _ in range(2)]
        cache.append_frame(i, [(key, key + 1) for key in keys], score_keys=keys,
                           frame_queries=[torch.ones(4, 32, device=device)] * 2,
                           rope_start=2 + 16 * i)
    cache.close_segment(None, scoring_strategy='frame_mean_query')
    for i in range(4):
        frame = cache.frames[i]
        cache.frames[i] = replace(frame, layers=tuple(
            entry.drop_low().shrink(8).shrink(8).shrink(8) for entry in frame.layers))
    cache.budget = cache.persistent_bytes() - 500
    return cache


class OffloadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_budget_archive_retrieval_snapshots_and_accounting(self):
        for granularity in ('layer', 'head'):
            for scope in ('segment', 'frame'):
                with self.subTest(granularity=granularity, scope=scope):
                    cache = make_cache(granularity, scope)
                    original = dict(cache.frames)
                    old_snapshot = cache.acquire_snapshot(1)
                    cache._enforce_budget()
                    stats = cache.audit_accounting()
                    self.assertEqual(set(cache.offloaded_frames), {0})
                    self.assertEqual(set(cache.frames), {1, 2, 3, 4})
                    self.assertIs(cache.frames[4], original[4])
                    self.assertLessEqual(stats['persistent_bytes'], cache.budget)
                    self.assertEqual(stats['retained_frames'], 5)
                    self.assertEqual(stats['resident_frames'], 4)
                    self.assertEqual(stats['offloaded_frames'], 1)
                    self.assertEqual(stats['counts']['evicted_frames'], 0)
                    self.assertEqual(stats['last_event']['offloaded'], 1)
                    self.assertGreater(stats['offloaded_tensor_bytes'], 0)
                    self.assertEqual(stats['offloaded_tensor_bytes'], tensor_storage_bytes(cache.offloaded_frames))
                    archived = cache.offloaded_frames[0]
                    self.assertEqual((archived.segment_id, archived.timestamp, archived.rope_start),
                                     (original[0].segment_id, original[0].timestamp, original[0].rope_start))
                    self.assertTrue(all(t.device.type == 'cpu' for t in tensors(archived)))
                    self.assertTrue(all(layer.bits == 2 and layer.low_k is None for layer in archived.layers))
                    states = cache.frame_states()['frames']
                    self.assertEqual(states[0]['status'], 'offloaded')
                    self.assertEqual(states[0]['counts'], [list(row) for row in cache._token_counts(original[0])])
                    cache.release_snapshot(1)
                    snapshot = cache.acquire_snapshot(2)
                    self.assertEqual([frame.frame_id for frame in snapshot.frames], list(range(5)))
                    q = torch.ones(1, 4, 2, 32)
                    for layer in range(2):
                        actual, ids = snapshot.retrieve_layer(layer, q, topk=5, headwise=True)
                        expected, reference_ids = old_snapshot.retrieve_layer(layer, q, topk=5, headwise=True)
                        self.assertEqual(ids, reference_ids)
                        self.assertIn(0, ids)
                        for a, e in zip(actual.heads, expected.heads):
                            torch.testing.assert_close(a.key, e.key, rtol=0, atol=0)
                            torch.testing.assert_close(a.value, e.value, rtol=0, atol=0)
                            torch.testing.assert_close(a.positions, e.positions, rtol=0, atol=0)
                    # Recent frames still consume the retrieval quota.
                    _, _, ids, _ = snapshot.retrieve_layer(0, q, topk=1)
                    self.assertEqual(ids, (4,))
                    cache.record_activation(2, [0, 0, 4])
                    self.assertEqual(cache.offloaded_frames[0].hits, 1)
                    self.assertEqual(snapshot.frames[0].hits, 0)
                    self.assertEqual(old_snapshot.frames[0].hits, 0)
                    cache.audit_accounting()
                    cache.release_snapshot(2)
                    cache.audit_accounting()
                    # Continue encoding and offloading without losing old archives.
                    cache.budget = 10**8
                    key = torch.ones(2, 16, 32)
                    cache.append_frame(5, [(key, key)] * 2, score_keys=[key] * 2,
                                       frame_queries=[torch.ones(4, 32)] * 2)
                    cache.close_segment(None, scoring_strategy='frame_mean_query')
                    cache.audit_accounting()
                    self.assertIn(0, cache.offloaded_frames)

    def test_offload_disabled_keeps_destructive_eviction(self):
        cache = make_cache(enabled=False)
        cache._enforce_budget()
        stats = cache.audit_accounting()
        self.assertFalse(stats['offload_enabled'])
        self.assertFalse(cache.offloaded_frames)
        self.assertEqual(stats['counts']['evicted_frames'], 1)
        self.assertEqual(stats['offloaded_bytes'], 0)
        self.assertNotIn(0, [frame.frame_id for frame in cache.acquire_snapshot(1).frames])

    def test_offload_preserves_lfu_then_fifo_order(self):
        cache = make_cache()
        frame = cache.frames[0]
        cache.frames[0] = replace(frame, hits=2)
        cache._enforce_budget()
        self.assertEqual(set(cache.offloaded_frames), {1})
        cache.budget = cache.persistent_bytes() - 500
        cache._enforce_budget()
        self.assertEqual(set(cache.offloaded_frames), {1, 2})
        cache.audit_accounting()

    def test_archived_frame_competes_with_resident_frames_for_topk(self):
        cache = make_cache()
        frame = cache.frames[0]
        cache.frames[0] = replace(frame, layers=tuple(replace(entry,
            representative=torch.full_like(entry.representative, 100)) for entry in frame.layers))
        cache._enforce_budget()
        self.assertIn(0, cache.offloaded_frames)
        snapshot = cache.acquire_snapshot(1)
        _, _, ids, _ = snapshot.retrieve_layer(0, torch.ones(1, 4, 1, 32), topk=2)
        self.assertEqual(ids, (0, 4))  # Best historical frame plus protected recent.
        cache.release_snapshot(1)
        cache.audit_accounting()

    def test_failed_cpu_copy_does_not_delete_frame(self):
        cache = make_cache()
        with patch.object(LayerFrame, 'to', side_effect=RuntimeError('copy failed')):
            with self.assertRaisesRegex(RuntimeError, 'copy failed'):
                cache._enforce_budget()
        self.assertIn(0, cache.frames)
        self.assertFalse(cache.offloaded_frames)
        cache.audit_accounting()

    def test_real_engine_can_answer_with_archived_frames_and_reset(self):
        from model.suprkv.onevision import OneVisionEngine
        from tests.test_suprkv_summarizer import tiny_model
        for granularity in ('layer', 'head'):
            with self.subTest(granularity=granularity):
                model, processor = tiny_model()
                config = StreamingConfig(device='cpu', budget_tokens=10000, recent_frames=1, topk=4,
                    group_size=8, scoring_strategy='frame_mean_query', scoring_scope='frame',
                    compression_granularity=granularity, offload_enabled=True, max_new_tokens=2)
                factory = lambda cfg: OneVisionEngine(cfg, model=model, processor=processor)
                with StreamingRuntime(config, factory) as runtime:
                    image = np.full((56, 56, 3), 128, dtype=np.uint8)
                    for i in range(4):
                        runtime.append_frame(image, i)
                    cache = runtime.model.cache
                    for i in range(3):
                        frame = cache.frames[i]
                        cache.frames[i] = replace(frame, layers=tuple(
                            entry.drop_low().shrink(8).shrink(8).shrink(8) for entry in frame.layers))
                    cache.budget = cache.persistent_bytes() - 500
                    cache._enforce_budget()
                    self.assertTrue(cache.offloaded_frames)
                    answer = runtime.ask('what?')
                    self.assertEqual(answer['hit_frames'], [0, 1, 2, 3])
                    self.assertEqual(cache.offloaded_frames[0].hits, 1)
                    cache.audit_accounting()
                    runtime.reset()
                    self.assertTrue(runtime.model.cache.offload_enabled)
                    self.assertFalse(runtime.model.cache.offloaded_frames)
                    self.assertEqual(runtime.memory_stats()['offloaded_tensor_bytes'], 0)


class OffloadConfigTests(unittest.TestCase):
    def test_config_cli_worker_forwarding_and_result_isolation(self):
        self.assertFalse(StreamingConfig().offload_enabled)
        parser, method = base.build_parser('suprkv'), SuPrKVMethod()
        required = ['--anno_path', 'unused.json', '--save_dir', 'unused']
        self.assertFalse(make_config(parser.parse_args(required)).offload_enabled)
        for strategy in ('summary', 'frame_mean_query'):
            args_base = required + ['--scoring-strategy', strategy]
            default = method.result_dir(parser.parse_args(args_base), 'rvs_ego')
            for flag in ('--offload', '--offload-enabled', '--offload_enabled'):
                args = parser.parse_args(args_base + [flag])
                forwarded = parser.parse_args(required + method.worker_arguments(args))
                self.assertTrue(make_config(forwarded).offload_enabled)
                directory = method.result_dir(args, 'rvs_ego')
                self.assertNotEqual(default, directory)
                self.assertIn('/offload_cpu', directory)
                self.assertFalse(make_config(parser.parse_args(args_base + [flag, '--no-offload'])).offload_enabled)
        for invalid in (1, 'true', None):
            with self.assertRaisesRegex(ValueError, 'offload_enabled'):
                StreamingConfig(offload_enabled=invalid)

    def test_observation_boolean_validation(self):
        dataset = SimpleNamespace(get=lambda _: dict(conversations=[dict(question='q')]))
        raw = dict(sample_id='sample', mode='compression')
        self.assertFalse(validate_params(raw, dataset)['offload_enabled'])
        self.assertTrue(validate_params(dict(raw, offload_enabled=True), dataset)['offload_enabled'])
        for invalid in ('false', 1, None):
            with self.assertRaisesRegex(ValueError, 'offload_enabled'):
                validate_params(dict(raw, offload_enabled=invalid), dataset)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for physical CPU/GPU transfers')
class OffloadCudaTests(unittest.TestCase):
    def test_pinned_archive_only_loads_selected_layer_and_matches_resident_retrieval(self):
        for granularity in ('layer', 'head'):
            cache = make_cache(granularity, device='cuda:0')
            reference = cache.acquire_snapshot(1)
            cache._enforce_budget()
            archive = cache.offloaded_frames[0]
            self.assertTrue(all(t.device.type == 'cpu' and t.is_pinned() for t in tensors(archive)))
            cache.release_snapshot(1)
            snapshot = cache.acquire_snapshot(2)
            loaded = []
            original_to = LayerFrame.to
            def record(entry, device, **kwargs):
                loaded.append(entry)
                return original_to(entry, device, **kwargs)
            q = torch.ones(1, 4, 2, 32, device='cuda:0')
            with patch.object(LayerFrame, 'to', record):
                snapshot.retrieve_layer(0, q, topk=1, headwise=True)
                self.assertFalse(loaded)
                actual, ids = snapshot.retrieve_layer(1, q, topk=5, headwise=True)
            self.assertEqual(len(loaded), 1)
            self.assertIs(loaded[0], archive.layers[1])
            expected, expected_ids = reference.retrieve_layer(1, q, topk=5, headwise=True)
            self.assertEqual(ids, expected_ids)
            for a, e in zip(actual.heads, expected.heads):
                torch.testing.assert_close(a.key, e.key, rtol=0, atol=0)
                torch.testing.assert_close(a.value, e.value, rtol=0, atol=0)
            self.assertNotIn(0, cache.frames)
            cache.audit_accounting()


if __name__ == '__main__':
    unittest.main()
