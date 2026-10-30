"""Frame-mean Q diagnostics: independent dense reference and HTTP contract."""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from observation.segments import TokenStateStore, frame_query_attention
from observation.server import ObservationService, handler_for, validate_params


class FrameQueryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.frames = [dict(index=i, token_start=2+i*3, token_end=5+i*3) for i in range(3)]
        self.result = dict(model=dict(layers=2, heads=4, kv_heads=2), frames=self.frames,
            sequence=dict(tokens_per_frame=3), segment_experiment=dict(frame_query_attention={}, segments=[
                dict(id=0, start_frame=0, end_frame=2, frame_indices=[0, 1]),
                dict(id=1, start_frame=2, end_frame=3, frame_indices=[2])]))
        torch.manual_seed(39)
        self.queries, self.keys = [], []
        for layer in range(2):
            # Includes prompt/suffix outliers which must not enter the means or scores.
            q = torch.randn(1, 13, 16).bfloat16()
            q[:, :2] = 1000; q[:, 11:] = -1000
            k = torch.randn(1, 13, 8)
            TokenStateStore(self.root, 'frame_query').frame_queries(layer, q, self.frames)
            TokenStateStore(self.root).video(layer, k, self.frames)
            self.queries.append(q[:, 2:11].float().reshape(3, 3, 4, 4).mean(1))
            self.keys.append(k[:, 2:11].reshape(3, 3, 2, 4))

    def reference(self, layer, start=0, end=2):
        q = self.queries[layer][start:end].double().permute(1, 0, 2)
        k = self.keys[layer][start:end].double().reshape(-1, 2, 4).permute(1, 0, 2).repeat_interleave(2, 0)
        logits = q @ k.transpose(-1, -2) / 2
        return logits.mean((0, 1)).numpy(), logits.softmax(-1).mean((0, 1)).numpy()

    def test_fp32_frame_pooling_and_dense_gqa_layer_aggregation(self):
        for layer in range(2):
            saved = np.load(self.root / f'frame_query_means/video-{layer}.npy')
            self.assertEqual(saved.dtype, np.float32)
            np.testing.assert_array_equal(saved, self.queries[layer].reshape(3, 16))
        for selected in ('0', '1', 'all'):
            reference = [self.reference(i) for i in (range(2) if selected == 'all' else [int(selected)])]
            expected = np.mean(reference, axis=0)
            pieces = [frame_query_attention(self.root, self.result, 0, i, selected) for i in range(2)]
            for measure, row in zip(('pre_softmax', 'post_softmax'), expected):
                np.testing.assert_allclose(np.concatenate([p[measure] for p in pieces]), row, atol=1e-12)
            self.assertEqual(pieces[0]['frame_masses'], pieces[1]['frame_masses'])
            self.assertAlmostEqual(pieces[0]['segment_mass'], 1.)
            self.assertLess(sum(pieces[0]['post_softmax']), 1.)
            self.assertEqual(pieces[1]['frame_token_start'], 5)
            self.assertEqual(pieces[0]['query_count'], 2)
            self.assertAlmostEqual(pieces[0]['pre_frame_mean'], np.mean(pieces[0]['pre_softmax']))
            json.dumps(pieces, allow_nan=False)
        single = frame_query_attention(self.root, self.result, 1, 2, '0')
        np.testing.assert_allclose(single['post_softmax'], self.reference(0, 2, 3)[1])

    def test_zero_queries_and_validation(self):
        for layer in range(2):
            np.save(self.root / f'frame_query_means/video-{layer}.npy', np.zeros((3, 16), np.float32))
        result = frame_query_attention(self.root, self.result, 0, 0, 'all')
        np.testing.assert_array_equal(result['pre_softmax'], [0, 0, 0])
        np.testing.assert_allclose(result['post_softmax'], [1/6]*3)
        for args in [(0, 2, 'all'), (-1, 0, 'all'), (0, -1, 'all'), (0, 0, '2'),
                     (0, 0, '../0'), (True, 0, '0')]:
            with self.assertRaises(ValueError):
                frame_query_attention(self.root, self.result, *args)
        for bad in (np.zeros((2, 16)), np.zeros((3, 15)), np.full((3, 16), np.nan)):
            np.save(self.root / 'frame_query_means/video-0.npy', bad)
            with self.assertRaises(ValueError):
                frame_query_attention(self.root, self.result, 0, 0, '0')
        (self.root / 'frame_query_means/video-0.npy').unlink()
        with self.assertRaisesRegex(ValueError, 'regenerate'):
            frame_query_attention(self.root, self.result, 0, 0, '0')
        del self.result['segment_experiment']['frame_query_attention']
        with self.assertRaisesRegex(ValueError, 'regenerate'):
            frame_query_attention(self.root, self.result, 0, 0, '0')

    def test_service_and_route_validate_completed_result(self):
        job = 'a'*24
        (self.root/job).mkdir()
        for name in ('token_keys', 'frame_query_means'):
            (self.root/name).rename(self.root/job/name)
        (self.root/job/'result.json').write_text(json.dumps(self.result))
        service = ObservationService.__new__(ObservationService)
        service.cache = self.root; service.similarity_lock = threading.Lock()
        handler = handler_for(service).__new__(handler_for(service))
        handler.send_json = Mock()
        handler.path = f'/api/frame-query-attention/{job}?segment=0&frame=0&layer=all'
        handler.do_GET()
        self.assertEqual(handler.send_json.call_args.args[0]['query_count'], 2)
        for query in ('segment=0&frame=2&layer=all', 'segment=0&frame=0&layer=all&frame=1',
                      'segment=0&frame=0&layer=all&rope=pre', 'segment=0&layer=all'):
            handler.path = f'/api/frame-query-attention/{job}?{query}'
            handler.do_GET()
            self.assertEqual(handler.send_json.call_args.args[1], 404)
        (self.root/job/'result.json').unlink()
        with self.assertRaises(FileNotFoundError):
            service.frame_query_attention(job, 0, 0, 'all')

    def test_strategy_validation_ignores_unused_summarizer(self):
        dataset = SimpleNamespace(get=lambda _: dict(conversations=[{}]))
        params = dict(sample_id='x', mode='compression', scoring_strategy='frame_mean_query',
                      summarizer_model_path='  /missing/checkpoint  ')
        actual = validate_params(params, dataset)
        self.assertEqual(actual['scoring_strategy'], 'frame_mean_query')
        self.assertIsNone(actual['summarizer_model_path'])
        for strategy in ('invalid', None, [], True):
            with self.assertRaises(ValueError):
                validate_params(dict(params, scoring_strategy=strategy), dataset)


if __name__ == '__main__':
    unittest.main()
