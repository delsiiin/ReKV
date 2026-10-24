"""Video window and question input checks; no model downloads."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2Model

from observation.engine import Cancelled, install_observer, question_inputs
from observation.metrics import (build_views, query_importance,
                                 validate_video_window, window_importance)
from observation.server import ObservationService, validate_params


class WindowAttentionTests(unittest.TestCase):
    def test_chunked_queries_match_dense_gqa_with_masks_and_cache_offsets(self):
        torch.manual_seed(81)
        q, k = torch.randn(1, 6, 51, 8), torch.randn(1, 2, 51, 8)
        selected = [19, 20, 28, 35, 44, 49]
        logits = q @ k.repeat_interleave(3, dim=1).transpose(-1, -2) * .3
        causal = torch.ones(51, 51, dtype=torch.bool).tril()
        for mask_heads in [1, 6]:
            mask = torch.ones(1, mask_heads, 51, 51, dtype=torch.bool)
            mask[..., 0] = False
            for additive in [False, True]:
                attention_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, -torch.inf) if additive else mask
                expected = logits.masked_fill(~causal, -torch.inf).masked_fill(~mask, -torch.inf)
                expected = expected.softmax(-1)[0, :, selected].mean((0, 1)).numpy()
                for offset in [0, 17]:
                    with self.subTest(mask_heads=mask_heads, additive=additive, offset=offset):
                        actual = query_importance(
                            q[:, :, offset:], k, selected, scaling=.3,
                            attention_mask=attention_mask[..., offset:, :], query_chunk_size=2)
                        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=2e-6)
                        self.assertAlmostEqual(float(actual.sum()), 1, places=6)
                        self.assertEqual(actual[-1], 0)  # Future key is never visible.

    def test_video_window_boundaries_and_legacy_history(self):
        torch.manual_seed(82)
        q, k = torch.randn(1, 4, 81, 8), torch.randn(1, 2, 81, 8)
        frames = [dict(index=i, token_start=5 + i * 38, token_end=43 + i * 38) for i in range(2)]
        dense = q @ k.repeat_interleave(2, dim=1).transpose(-1, -2) / np.sqrt(8)
        dense.masked_fill_(~torch.ones(81, 81, dtype=torch.bool).tril(), -torch.inf)
        for window in [1, 32, 40, 76]:
            with self.subTest(window=window):
                validate_video_window(window, 76)
                score = window_importance(q, k, window=window)
                expected = dense.softmax(-1)[0, :, -window:].mean((0, 1)).numpy()
                np.testing.assert_allclose(score, expected, atol=1e-7, rtol=2e-6)
                view = build_views([score], frames, 81 - window)[0]
                self.assertEqual(sum(f['window_tokens'] for f in view['frames']), window)
        for bad in [0, -1, 1.5, True, '32', 77]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_video_window(bad, 76)
        np.testing.assert_array_equal(window_importance(q, k), window_importance(q, k, window=32))

    def test_chunk_cancellation_cleans_up_observer(self):
        model = Qwen2Model(Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1))
        model.set_attn_implementation('sdpa')
        calls = []

        def cancel():
            calls.append(1)
            if len(calls) > 1:  # Cancel within a query chunk, after entering the hook.
                raise Cancelled('cancelled')

        cleanup = install_observer(model, lambda *_: self.fail('Partial scores must not be captured'),
                                   window=37, check_cancel=cancel)
        try:
            with self.assertRaises(Cancelled):
                model(torch.zeros(1, 40, dtype=torch.long))
        finally:
            cleanup()
        self.assertEqual(model.config._attn_implementation, 'sdpa')
        self.assertFalse(hasattr(model.layers[0].self_attn, '_rvs_observer'))


class WindowParameterTests(unittest.TestCase):
    def test_validation_and_window_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data/movie/videos').mkdir(parents=True)
            (root / 'model').mkdir()
            (root / 'model/config.json').write_text('{}')
            (root / 'data/movie/movienet_oe.json').write_text(json.dumps([
                dict(video_id='sample', video_path='sample.npy', conversations=[dict(question='q')])]))
            np.save(root / 'data/movie/videos/sample.npy', np.zeros((2, 2, 2, 3), dtype=np.uint8))
            args = SimpleNamespace(data_root=root / 'data', cache_dir=root / 'cache',
                                   model_path=root / 'model', device='cpu', dtype='float32')
            service = ObservationService(args)
            try:
                default = validate_params(dict(sample_id='movie:0'), service.dataset)
                self.assertEqual(default['video_window_size'], 32)
                self.assertEqual(service.cache_key(default), service.cache_key(dict(default, video_window_size=32)))
                self.assertNotEqual(service.cache_key(default), service.cache_key(dict(default, video_window_size=64)))
                for bad in [0, -1, 2.5, True, False, '32', None]:
                    with self.subTest(bad=bad), self.assertRaises(ValueError):
                        validate_params(dict(default, video_window_size=bad), service.dataset)
            finally:
                service.close()


class QuestionTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'model_zoo/llava-onevision-qwen2-7b-ov-hf'
        if not (path / 'tokenizer.json').is_file():
            raise unittest.SkipTest('Local tokenizer is not installed')
        cls.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    def test_question_suffix_matches_answer_prompt(self):
        for question in ['Why?', '这一帧中有什么？', 'Why? ' * 40, '  What happens next?\n']:
            with self.subTest(question=question[:20]):
                ids = question_inputs(self.tokenizer, question)
                expected = self.tokenizer('\n' + question + '<|im_end|><|im_start|>assistant\n')['input_ids']
                self.assertEqual(ids[0].tolist(), expected)
        for question in ['', ' \n', None]:
            with self.assertRaises(ValueError):
                question_inputs(self.tokenizer, question)


if __name__ == '__main__':
    unittest.main()
