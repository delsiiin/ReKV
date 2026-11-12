"""Runtime tests independent of the optional upstream StreamKV checkout."""
from contextlib import ExitStack, contextmanager, nullcontext, redirect_stderr
from io import StringIO
import inspect
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration, Qwen2Config

from model.streamkv import cache
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


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_cli_worker_round_trip_and_validation(self):
        method = get_method('streamkv')
        for flag in ('--budget_tokens', '--budget-tokens'):
            args = run_eval.build_parser().parse_args([
                '--method', 'streamkv', '--dataset', 'mlvu', flag, '8192'])
            with self.assertRaisesRegex(ValueError, 'budget_tokens.*not supported'):
                validate_arguments(args, method)
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                base.build_parser('streamkv').parse_args([
                    '--anno_path', 'unused', '--save_dir', 'unused', flag, '8192'])
        with self.assertRaises(TypeError):
            StreamKVConfig(budget_tokens=8192)
        args = run_eval.build_parser().parse_args([
            '--method', 'streamkv', '--dataset', 'mlvu', '--compression_ratio', '.6',
            '--compress_temp', '1'])
        validate_arguments(args, method)
        worker = base.build_parser('streamkv').parse_args([
            '--anno_path', 'unused', '--save_dir', 'unused', *method.worker_arguments(args)])
        self.assertEqual(method.config(worker), method.config(args))
        self.assertEqual(method.config(worker).compression_ratio, .6)
        self.assertNotIn('--budget_tokens', method.worker_arguments(args))
        self.assertNotIn('_budget', method.result_dir(args, 'mlvu'))
        self.assertNotIn('budget_tokens', method.analyzer_kwargs(args)['record_metadata'])

    @torch.inference_mode()
    def test_ratio_mode_across_segments_retrieval_and_reset(self):
        with cpu_cache_execution():
            for summary in (False, True):
                for ratio in (1., .6):
                    with self.subTest(summary=summary, ratio=ratio):
                        config = StreamKVConfig(n_local=24, retrieve_size=3, retrieve_local_size=1,
                                                use_segment_summary=summary,
                                                compression_ratio=ratio, compress_temp=1)
                        runtime = tiny_runtime(config)
                        runtime.encode_init_prompt()
                        for _ in range(4):
                            runtime._encode_video_chunk(torch.randn(1, 16, 32))
                            for layer in runtime.kv_cache:
                                self.assertIs(type(layer), cache.ContextManager)
                                self.assertEqual(len(layer.global_blocks[0]), layer.num_global_block)
                                self.assertEqual(len(layer.block_k[0]), layer.num_global_block)
                            self.assertTrue(runtime.question_answering(
                                dict(question='why', prompt='what'), max_new_tokens=2))
                        runtime.clear_cache()
                        self.assertIsNone(runtime.kv_cache)
                        runtime.encode_init_prompt()
                        runtime._encode_video_chunk(torch.randn(1, 16, 32))
                        for layer in runtime.kv_cache:
                            self.assertLessEqual(layer.num_global_block, 4 + summary)


if __name__ == '__main__':
    unittest.main()
