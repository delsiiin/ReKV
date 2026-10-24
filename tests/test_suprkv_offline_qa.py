"""Exercise offline QA overrides through the real runtime and a tiny CPU decoder."""
from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import Qwen2Config, Qwen2Model

from model.llava_onevision_rekv import LlavaOneVision_ReKV
from model.patch import patch_hf
from model.suprkv.attention import streaming_attention_forward
from model.suprkv.cache import StreamingKVCache
from model.suprkv.onevision import OneVisionEngine
from model.suprkv.runtime import StreamingConfig, StreamingRuntime
from video_qa.rekv_offline_vqa import ReKVOfflineVQA
from video_qa.suprkv_offline_vqa import SuPrKVOfflineVQA
from video_qa.suprkv_stream_vqa import SuPrKVStreamVQA


class RecordingTokenizer:
    eos_token_id = 31

    def __init__(self):
        self.texts = []
        self.decode = Mock(return_value=' B)\nchosen ')

    def __call__(self, text, **kwargs):
        self.texts.append(text)
        return SimpleNamespace(input_ids=torch.tensor([[3 + ord(c) % 28 for c in text]]))


class OfflineQAParityTests(unittest.TestCase):
    @torch.inference_mode()
    def setUp(self):
        torch.manual_seed(19)
        self.config = StreamingConfig(device='cpu', budget_tokens=10000, topk=3,
                                      group_size=8, max_new_tokens=7)
        engine = object.__new__(OneVisionEngine)
        engine.config = asdict(self.config)
        engine.device, engine.frame_tokens = torch.device('cpu'), 4
        engine.decoder = Qwen2Model(Qwen2Config(
            vocab_size=32, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)).half().eval()
        patch_hf(engine.decoder, attention_factory=streaming_attention_forward,
                 n_local=8, n_init=2, fattn=False)
        engine.model = SimpleNamespace(lm_head=torch.nn.Linear(32, 32, bias=False).half().eval())
        engine.tokenizer = RecordingTokenizer()
        initial = engine.decoder(input_ids=torch.tensor([[1, 2]]), use_cache=True)
        sink = [(k[0], v[0]) for k, v in initial.past_key_values]
        engine.cache = StreamingKVCache(sink, 10000, 4, group_size=8)
        for index in range(4):
            key = torch.randn(2, 4, 8).half()
            engine.cache.append_frame(index, [(key, key)] * 2, score_keys=[key, key])
        engine.cache.close_segment([torch.ones(2, 4, 8)] * 2)
        self.engine = engine
        self.runtime = StreamingRuntime(self.config, model_factory=lambda config: engine)
        self.runtime.last_timestamp = 3.  # The fixture already contains four encoded frames.
        self.addCleanup(self.runtime.close)
        sample = dict(video_id='v', conversations=[dict(question='q', answer='a')])
        args = dict(anno=[sample], save_dir='unused', sample_fps=1)
        self.offline = SuPrKVOfflineVQA(qa_model=self.runtime, **args)
        self.stream = SuPrKVStreamVQA(qa_model=self.runtime, **args)
        self.reference_model = Mock()
        self.reference_model.get_prompt.side_effect = lambda q, mc=False: (
            LlavaOneVision_ReKV.get_prompt(None, q, mc=mc))
        self.reference_model.question_answering.return_value = ' B)\nchosen '
        self.reference = ReKVOfflineVQA(qa_model=self.reference_model, **args)

    def test_offline_matches_rekv_and_leaves_stream_defaults_and_video_cache_intact(self):
        engine = self.engine
        config_before = dict(engine.config)
        cache_version = engine.cache.version
        payloads = {i: frame.layers for i, frame in engine.cache.frames.items()}
        limits, windows = [], []

        def generate(decoder, lm_head, output, limit, stop_tokens, check_cancel):
            limits.append(limit)
            windows.append([layer.self_attn._streaming_n_local for layer in decoder.layers])
            return [3, 31], 'eos'

        with patch('model.suprkv.onevision.greedy_token_ids', side_effect=generate):
            before = self.stream.video_open_qa('stream question')
            mc = self.offline.video_close_qa('which one?', ['first', 'second'], 'B')
            self.assertEqual(mc, self.reference.video_close_qa('which one?', ['first', 'second'], 'B'))
            reference_mc = self.reference_model.question_answering.call_args.args[0]
            opened = self.offline.video_open_qa('describe the video')
            self.assertEqual(opened, self.reference.video_open_qa('describe the video'))
            reference_open = self.reference_model.question_answering.call_args.args[0]
            after = self.stream.video_open_qa('stream question')

        self.assertEqual(limits, [7, 16, 1024, 7])
        self.assertEqual(windows, [[3 * 4 + limit + 1024] * 2 for limit in limits])
        self.assertEqual(engine.tokenizer.texts[2:6], [
            reference_mc['question'], reference_mc['prompt'],
            reference_open['question'], reference_open['prompt']])
        self.assertEqual(reference_mc['question'], 'which one?')
        self.assertTrue(reference_mc['prompt'].endswith('assistant\nBest option: ('))
        self.assertEqual(engine.tokenizer.texts[:2], engine.tokenizer.texts[-2:])
        self.assertEqual(before, after)
        self.assertEqual(before['pred_answer'], ' B)chosen ')
        self.assertEqual(engine.config, config_before)
        self.assertEqual(self.runtime.config, self.config)
        self.assertEqual(engine.cache.version, cache_version)
        self.assertEqual(set(engine.cache.frames), set(payloads))
        for index, layers in payloads.items():
            self.assertIs(engine.cache.frames[index].layers, layers)
        self.assertFalse(engine.cache.snapshots)
        engine._set_window('encode')
        self.assertTrue(all(layer.self_attn._streaming_n_local == 8 for layer in engine.decoder.layers))

    def test_invalid_overrides_and_external_retrieval_do_not_close_runtime_or_acquire_snapshots(self):
        for kwargs in ({'prompt': ''}, {'prompt': 5}, {'max_new_tokens': 0},
                       {'max_new_tokens': -1}, {'max_new_tokens': 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.runtime.ask('question', **kwargs)
        with self.assertRaises(NotImplementedError):
            self.runtime.question_answering({'question': 'q', 'prompt': 'p'}, retrieved_indices=[0])
        self.assertEqual(self.runtime.question_id, 0)
        self.assertFalse(self.runtime.closed)
        self.assertFalse(self.engine.cache.snapshots)


if __name__ == '__main__':
    unittest.main()
