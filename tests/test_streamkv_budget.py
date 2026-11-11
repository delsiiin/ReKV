"""Budget-mode tests independent of the optional upstream StreamKV checkout."""
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
import inspect
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration, Qwen2Config

from model.streamkv import cache
from model.streamkv.abstract_streamkv import Abstract_StreamKV
from model.streamkv.budget import BudgetContextManager
from model.streamkv.config import StreamKVConfig
from model.streamkv.runtime import StreamKVRuntime
from video_qa import base, run_eval
from video_qa.methods import get_method, validate_arguments


@contextmanager
def cpu_cache_execution():
    """Replace CUDA allocation/stream plumbing only; exercise real cache logic."""
    with ExitStack() as stack:
        for name, value in [('stream', lambda _: nullcontext()), ('Event', Mock),
                            ('current_stream', lambda: None), ('empty_cache', lambda: None),
                            ('ipc_collect', lambda: None)]:
            stack.enter_context(patch.object(torch.cuda, name, value))
        source = textwrap.dedent(inspect.getsource(cache.ContextManager.init))
        source = source.replace('        assert _t.is_cuda\n', '')
        namespace = dict(vars(cache))
        exec(compile(source, '<CPU cache init>', 'exec'), namespace)
        stack.enter_context(patch.object(cache.ContextManager, 'init', namespace['init']))

        def allocate(self, num_units, unit_size, dtype):
            self.num_units, self.unit_size, self.dtype = num_units, unit_size, dtype
            self.data = torch.empty((num_units, unit_size), dtype=dtype)
            self.idle_set = set(range(num_units))

        stack.enter_context(patch.object(cache.CudaCache, '__init__', allocate))
        yield


class Tokenizer:
    eos_token_id = None

    def __call__(self, text, **kwargs):
        ids = [3 + ord(c) % 27 for c in text[:3]]
        return SimpleNamespace(input_ids=torch.tensor([ids]) if kwargs.get('return_tensors') else ids)

    def decode(self, tokens, **kwargs):
        return ' '.join(map(str, tokens))


def tiny_runtime(config):
    torch.manual_seed(24)
    model_config = LlavaOnevisionConfig(
        text_config=Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=48,
                               num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2),
        vision_config=dict(model_type='siglip_vision_model', hidden_size=32, intermediate_size=48,
                           num_hidden_layers=2, num_attention_heads=4, image_size=8, patch_size=2))
    model = LlavaOnevisionForConditionalGeneration(model_config).eval()
    return StreamKVRuntime(model, SimpleNamespace(tokenizer=Tokenizer()), config,
                           n_frame_tokens=4, fattn=False, async_global_stream=False, pin_memory=False)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_cli_worker_round_trip_and_validation(self):
        method = get_method('streamkv')
        for flag in ('--budget_tokens', '--budget-tokens'):
            args = run_eval.build_parser().parse_args([
                '--method', 'streamkv', '--dataset', 'mlvu', flag, '8192'])
            validate_arguments(args, method)
            worker = base.build_parser('streamkv').parse_args([
                '--anno_path', 'unused', '--save_dir', 'unused', *method.worker_arguments(args)])
            self.assertEqual(method.config(worker), method.config(args))
            self.assertIn('_budget8192/', method.result_dir(args, 'mlvu'))
            self.assertEqual(method.analyzer_kwargs(args)['record_metadata']['budget_tokens'], 8192)
        for budget in (0, -1, 195):
            args.budget_tokens = budget
            with self.assertRaisesRegex(ValueError, 'budget_tokens'):
                validate_arguments(args, method)
        for budget in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                StreamKVConfig(budget_tokens=budget)
        args.budget_tokens = None
        self.assertNotIn('--budget_tokens', method.worker_arguments(args))
        self.assertNotIn('_budget', method.result_dir(args, 'mlvu'))

    def test_archive_cap_across_segments_retrieval_and_reset(self):
        with cpu_cache_execution():
            for summary in (False, True):
                for budget in (4, 11, 100):
                    with self.subTest(summary=summary, budget=budget):
                        config = StreamKVConfig(n_local=24, retrieve_size=3, retrieve_local_size=1,
                                                use_segment_summary=summary, budget_tokens=budget)
                        runtime = tiny_runtime(config)
                        runtime.encode_init_prompt()
                        for _ in range(4):
                            runtime._encode_video_chunk(torch.randn(1, 16, 32))
                            for layer in runtime.kv_cache:
                                self.assertIsInstance(layer, BudgetContextManager)
                                self.assertLessEqual(layer.num_global_block * 4, budget)
                                self.assertEqual(len(layer.global_blocks[0]), layer.num_global_block)
                                self.assertEqual(len(layer.block_k[0]), layer.num_global_block)
                                self.assertEqual([i for chunk in layer.chunk2block[0] for i in chunk],
                                                 list(range(layer.num_global_block)))
                            self.assertTrue(runtime.question_answering(
                                dict(question='why', prompt='what'), max_new_tokens=2))
                        runtime.clear_cache()
                        runtime.encode_init_prompt()
                        runtime._encode_video_chunk(torch.randn(1, 4, 32))
                        for layer in runtime.kv_cache:
                            self.assertEqual(layer.num_global_block, min(1 + summary, budget // 4))

    def test_global_scoring_selection_and_cached_index_remap(self):
        # Two batch items retain different blocks, including a cached survivor
        # that changes index and an evicted block that must release its GPU slot.
        layer = BudgetContextManager.__new__(BudgetContextManager)
        layer.block_size, layer.num_units, layer.num_global_block = 4, 2, 4
        layer.global_blocks = [[Mock() for _ in range(4)] for _ in range(2)]
        original = [list(blocks) for blocks in layer.global_blocks]
        layer.cached_blocks = [{0: 1, 2: 3}, {1: 2, 3: 4}]
        layer.chunk2block = [[[0, 1], [2, 3]], [[0, 1], [2, 3]]]
        layer.block_k = [cache.VectorTensor(2, torch.float32, 'cpu') for _ in range(2)]
        vectors = torch.tensor([[1., 0.], [0., 1.], [3., 0.], [0., 4.]])
        for vector in layer.block_k:
            vector.append(vectors)
        layer.unit_size, layer.dim_head, layer.use_segment_summary = 1, 2, 'False'
        layer.calculate_score(8, torch.tensor([[[[1., 0.]]], [[[0., 1.]]]]))
        torch.testing.assert_close(layer.archive_compression_score,
                                   torch.tensor([[1., 0., 3., 0.], [0., 1., 0., 4.]]))
        layer.compress_to_budget(4)
        self.assertEqual(layer.num_global_block, 1)
        for u, retained in enumerate((2, 3)):
            self.assertEqual(layer.global_blocks[u], [original[u][retained]])
            torch.testing.assert_close(layer.block_k[u].get_data(), vectors[retained:retained + 1])
            self.assertEqual(layer.chunk2block[u], [[0]])
            original[u][retained].offload.assert_not_called()
        original[0][0].offload.assert_called_once()
        original[1][1].offload.assert_called_once()
        self.assertEqual(layer.cached_blocks, [{0: 3}, {0: 4}])

    @torch.inference_mode()
    def test_default_mode_matches_reference_chunk_encoding(self):
        with cpu_cache_execution():
            for ratio in (1., .6):
                config = StreamKVConfig(n_local=24, compression_ratio=ratio, compress_temp=1)
                runtime = tiny_runtime(config)
                reference = tiny_runtime(config)
                reference._encode_video_chunk = Abstract_StreamKV._encode_video_chunk.__get__(reference)
                for model in (runtime, reference):
                    model.encode_init_prompt()
                for _ in range(2):
                    features = torch.randn(1, 16, 32)
                    runtime._encode_video_chunk(features)
                    reference._encode_video_chunk(features)
                for actual, expected in zip(runtime.kv_cache, reference.kv_cache):
                    self.assertIs(type(actual), cache.ContextManager)
                    torch.testing.assert_close(actual.block_k[0].get_data(), expected.block_k[0].get_data())
                    torch.testing.assert_close(actual.local_k, expected.local_k)
                    for a, b in zip(actual.global_blocks[0], expected.global_blocks[0]):
                        torch.testing.assert_close(a.cpu_data[0], b.cpu_data[0])

    def test_budget_overrides_ratio_and_temperature(self):
        with cpu_cache_execution():
            config = StreamKVConfig(n_local=24, budget_tokens=8)
            first = tiny_runtime(config)
            second = tiny_runtime(replace(config, compression_ratio=.6, compress_temp=1))
            first.encode_init_prompt()
            second.encode_init_prompt()
            features = torch.randn(1, 16, 32)
            first._encode_video_chunk(features)
            second._encode_video_chunk(features)
            for a, b in zip(first.kv_cache, second.kv_cache):
                torch.testing.assert_close(a.block_k[0].get_data(), b.block_k[0].get_data())


if __name__ == '__main__':
    unittest.main()
