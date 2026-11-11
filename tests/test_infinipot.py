"""Differential tests against the supplied InfiniPot-V, with tiny real Qwen."""
import ast
import copy
import csv
import importlib.util
import re
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from transformers import DynamicCache, Qwen2_5_VLForConditionalGeneration

from model.infinipot.cache import process_kv_cache
from model.infinipot.config import InfiniPotConfig
from model.infinipot.runtime import InfiniPotRuntime
from model.infinipot.rope import source_rope_index
from model.infinipot.video import frame_indices, load_video
from video_qa import base, run_eval
from video_qa.methods import get_method, validate_arguments
from video_qa.infinipot_offline_vqa import InfiniPotOfflineVQA
from video_qa.infinipot_stream_vqa import InfiniPotStreamVQA
from test_qwen2_5_vl import tiny_config
from test_qwen2_5_vl_native import native_processor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'InfiniPot-V'


def reference_core():
    """Compile the actual reference methods without importing datasets/GPU deps."""
    spec = importlib.util.spec_from_file_location('infinipot_reference_cache', SOURCE / 'kvcache_utils.py')
    cache = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cache)
    tree = ast.parse((SOURCE / 'qwen_inference_ovu.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    names = ('_visual_encoding_block', '_block_wise_prefill', '_generation_stage', 'block_process')
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = dict(torch=torch, DynamicCache=DynamicCache, process_kv_cache=cache.process_kv_cache)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'reference_core', 'exec'), scope)
    ref = scope['OfflineVideoEval']()
    ref._print = lambda message: None
    ref.MAX_GEN_TOKENS = 3
    ref.DEFAULT_EOS_TOKEN_ID = 151645
    return ref, cache.process_kv_cache


def processor():
    p = native_processor()
    p.chat_template = (
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' }}"
        "{% for item in message['content'] %}{% if item['type']=='video' %}"
        "{{ '<|vision_start|><|video_pad|><|vision_end|>' }}"
        "{% else %}{{ item['text'] }}{% endif %}{% endfor %}{{ '<|im_end|>\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
    return p


def tiny_runtime(config=None):
    model = Qwen2_5_VLForConditionalGeneration(tiny_config()).eval()
    model.generation_config.pad_token_id = 62
    model.generation_config.do_sample = False
    return InfiniPotRuntime(model, processor(), config, max_new_tokens=3)


@unittest.skipUnless(SOURCE.exists(), 'Differential tests require the supplied InfiniPot-V checkout')
class ReferenceParityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_cache_source_is_verbatim_and_all_modes_match(self):
        self.assertEqual((SOURCE / 'kvcache_utils.py').read_bytes(),
                         (ROOT / 'model/infinipot/cache.py').read_bytes())
        _, reference = reference_core()
        model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=[None] * 4)),
                                config=SimpleNamespace(height_width=2))
        for method in ('uniform', 'swa', 'infinipot-v'):
            for pooling in (False, True):
                for ratio in (0., .5, 1.):
                    with self.subTest(method=method, pooling=pooling, ratio=ratio):
                        cache = DynamicCache()
                        for layer in range(4):
                            cache.update(torch.randn(1, 3, 3 + 32 * 6, 8),
                                         torch.randn(1, 3, 3 + 32 * 6, 8), layer)
                        expected = copy.deepcopy(cache)
                        kwargs = dict(model=model, system_size=3, inst_size=5, token_per_frame=6,
                                      compress_frame_num=24, method=method, tar_ratio=ratio,
                                      adaptive_pooling=pooling, query_ratio=.25)
                        reference(expected, **kwargs)
                        prefix = cache.layers[0].keys[:, :, :3].clone()
                        process_kv_cache(cache, **kwargs)
                        self.assertEqual(cache.get_seq_length(), 147)
                        torch.testing.assert_close(cache.layers[0].keys[:, :, :3], prefix, rtol=0, atol=0)
                        for actual, wanted in zip(cache.layers, expected.layers):
                            torch.testing.assert_close(actual.keys, wanted.keys, rtol=0, atol=0)
                            torch.testing.assert_close(actual.values, wanted.values, rtol=0, atol=0)

    @torch.inference_mode()
    def test_real_qwen_block_cache_positions_and_generation_match_source(self):
        runtime = tiny_runtime()
        ref, _ = reference_core()
        ref.__dict__.update({k: getattr(runtime, k) for k in (
            'block_size', 'compress_frame_num', 'compression_method', 'tar_ratio', 'query_ratio',
            'adaptive_pooling', 'per_frame', 'processor')})
        ref.model = copy.deepcopy(runtime.model)
        # Sole HF API shim for the unmodified source's visual encoder.
        get_features = ref.model.get_video_features
        ref.model.get_video_features = lambda *a, **kw: get_features(*a, **kw).pooler_output
        for frames in (33, 40, 41, 49):
            with self.subTest(grid_frames=frames):
                runtime.set_video(torch.randint(0, 256, (frames * 2, 3, 8, 8), dtype=torch.uint8))
                inputs = runtime.prepare_inputs('describe')
                ids = inputs['input_ids']
                indices = (ids[0] == 61).nonzero().flatten()
                system = int(indices[0])
                inst = ids.shape[1] - int(indices[-1]) - 1
                tpf = indices.numel() // frames
                caches, traces = [], []
                for runner in (runtime, ref):
                    trace = []
                    def record(module, args, kwargs):
                        trace.append((kwargs['position_ids'].clone(), kwargs['past_key_values'].get_seq_length()))
                    handle = runner.model.register_forward_pre_hook(record, with_kwargs=True)
                    try:
                        result = runner._block_wise_prefill(inputs, ids, DynamicCache(), system, inst,
                                                           tpf, indices.numel(), frames)
                    finally:
                        handle.remove()
                    caches.append(result)
                    traces.append(trace)
                self.assertEqual(len(traces[0]), len(traces[1]))
                for actual, expected in zip(traces[0], traces[1]):
                    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                    self.assertEqual(actual[1], expected[1])
                self.assertEqual(caches[0][0].get_seq_length(), system + 32 * tpf)
                for actual, expected in zip(caches[0][0].layers, caches[1][0].layers):
                    torch.testing.assert_close(actual.keys, expected.keys, rtol=0, atol=0)
                    torch.testing.assert_close(actual.values, expected.values, rtol=0, atol=0)
                gen_traces = []
                answers = []
                for runner, (cache, positions) in zip((runtime, ref), caches):
                    trace = []
                    def record(module, args, kwargs):
                        trace.append(kwargs['position_ids'].clone())
                    handle = runner.model.register_forward_pre_hook(record, with_kwargs=True)
                    try:
                        answers.append(runner._generation_stage(inputs, cache, positions))
                    finally:
                        handle.remove()
                    # Source starts AFTER the full prompt, including the suffix.
                    self.assertEqual(int(trace[0][0, 0, 0]), int(positions[0, 0, -1]) + 1)
                    gen_traces.append(trace)
                self.assertEqual(answers[0], answers[1])
                for a, b in zip(*gen_traces):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)


class RuntimeAndDatasetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_reference_rope_text_follows_max_video_position(self):
        runtime = tiny_runtime()
        runtime.model.config.vision_config.tokens_per_second = 25
        ids = torch.tensor([[1, 58] + [61] * 12 + [59, 4, 5]])
        positions, delta = source_rope_index(runtime.model.model, ids, video_grid_thw=torch.tensor([[3, 4, 4]]))
        self.assertEqual(positions[0, 0, 10:14].tolist(), [52] * 4)
        self.assertEqual(positions[0, 0, -3:].tolist(), [53, 54, 55])
        self.assertEqual(int(delta), 56 - ids.shape[1])

    def test_standard_block_dispatch_limits_and_question_isolation(self):
        runtime = tiny_runtime()
        runtime.MAX_GEN_TOKENS = 2
        for frames, blocking in ((4, False), (64, False), (66, True)):
            runtime.set_video(torch.randint(0, 256, (frames, 3, 8, 8), dtype=torch.uint8))
            with patch.object(runtime, 'block_process', wraps=runtime.block_process) as block, \
                    patch.object(runtime.model, 'generate', wraps=runtime.model.generate) as standard:
                first = runtime.question_answering({'prompt': 'describe'}, max_new_tokens=16)
                self.assertEqual(block.called, blocking)
                self.assertEqual(standard.called, not blocking)
                if standard.called:
                    self.assertEqual(standard.call_args.kwargs['max_new_tokens'], 2)
            runtime.question_answering({'prompt': 'different question'})
            self.assertEqual(first, runtime.question_answering({'prompt': 'describe'}))
            self.assertIsNone(runtime.model.model.rope_deltas)
        runtime.reset()
        runtime.question_answering({'prompt': 'text only'})
        with patch.object(runtime, 'generate', side_effect=RuntimeError('failure')), self.assertRaises(RuntimeError):
            runtime.question_answering({'prompt': 'failure'})
        self.assertIsNone(runtime.model.model.rope_deltas)
        runtime.close()

    def test_sampling_cap_and_no_future_frames_in_npy_prefix(self):
        self.assertEqual(frame_indices(100, 10, 2, 768).tolist(), torch.linspace(0, 99, 20).round().long().tolist())
        self.assertEqual(len(frame_indices(100000, 25, 2, 768)), 768)
        self.assertEqual(frame_indices(1, 1).tolist(), [0, 0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'movie.npy'
            pixels = np.stack([np.full((28, 28, 3), i * 20, dtype=np.uint8) for i in range(10)])
            np.save(path, pixels)
            video = load_video(path, end_time=4, max_frames=768)
            self.assertEqual(len(video), 4)
            self.assertEqual([int(frame[0, 0, 0]) for frame in video], [0, 20, 40, 60])
            self.assertIsNone(load_video(path, end_time=0))
            self.assertEqual(len(load_video(path, end_time=1)), 2)

    def test_all_dataset_routes_and_argument_roundtrip(self):
        method = get_method('infinipot')
        for dataset, (_, mode, _) in run_eval.DATASETS.items():
            args = run_eval.build_parser().parse_args([
                '--method', 'infinipot', '--model', 'qwen2_5_vl_7b', '--dataset', dataset])
            validate_arguments(args, method)
            self.assertEqual(args.sample_fps, 2.)
            with patch.object(run_eval, 'load_samples', return_value=[{}]), \
                    patch.object(run_eval.multiprocessing, 'Process', return_value=Mock(exitcode=0)) as launch, \
                    patch.object(run_eval, 'exec'):
                run_eval.evaluate(args, dataset)
            command = launch.call_args.kwargs['args'][0]
            self.assertEqual(command[2], f'video_qa.infinipot_{mode}_vqa')
            worker = base.build_parser('infinipot').parse_args(command[3:])
            validate_arguments(worker, method)
            self.assertEqual(method.config(worker), InfiniPotConfig())
        for method_name in ('rekv', 'native', 'suprkv', 'streamkv'):
            args = run_eval.build_parser().parse_args(['--method', method_name, '--dataset', 'mlvu'])
            validate_arguments(args, get_method(method_name))
            self.assertEqual(args.sample_fps, 1.)

    def test_validation_and_reject_foreign_options(self):
        for kwargs in (dict(block_size=24), dict(query_ratio=-.1), dict(tar_ratio=2), dict(max_frames_num=1)):
            with self.assertRaises(ValueError):
                InfiniPotConfig(**kwargs)
        with self.assertRaisesRegex(ValueError, 'TaR'):
            InfiniPotConfig(compress_frame_num=1).validate_budget(128)
        for flags in (['--model', 'video_llava_7b'], ['--model', 'qwen2_5_vl_7b', '--retrieve_size', '32']):
            args = run_eval.build_parser().parse_args(['--method', 'infinipot', '--dataset', 'mlvu', *flags])
            with self.assertRaises(ValueError):
                validate_arguments(args, get_method('infinipot'))

    def test_dynamic_tail_budget_does_not_reject_valid_source_run(self):
        config = InfiniPotConfig(compress_frame_num=1)
        # A 33-grid video retains 31 grids at its only compression, not 1.
        config.validate_budget(4, total_frames=33)
        with self.assertRaisesRegex(ValueError, 'TaR'):
            config.validate_budget(4, total_frames=64)
        runtime = tiny_runtime(config)
        runtime.set_video(torch.randint(0, 256, (66, 3, 8, 8), dtype=torch.uint8))
        actual = runtime.question_answering({'prompt': 'describe'})
        with torch.inference_mode():
            expected = runtime.block_process(runtime.prepare_inputs('describe'))
        self.assertEqual(actual, expected)
        # Source never consumes query_ratio when TaR is disabled.
        InfiniPotConfig(tar_ratio=0, query_ratio=0).validate_budget(4, total_frames=64)
        with self.assertRaisesRegex(ValueError, 'TaR'):
            InfiniPotConfig(query_ratio=0).validate_budget(4, total_frames=64)

    @unittest.skipUnless(SOURCE.exists(), 'Requires the supplied InfiniPot-V checkout')
    def test_answer_extraction_matches_source_for_actual_choice_count(self):
        tree = ast.parse((SOURCE / 'dataset_utils.py').read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'extract_answer')
        scope = dict(re=re)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'source_answers', 'exec'), scope)
        flow = InfiniPotOfflineVQA.__new__(InfiniPotOfflineVQA)
        flow.choice_letters = list('ABCDEFGH')
        for count, dataset in ((4, 'mlvu'), (5, 'egoschema')):
            for answer in ('Final answer: D', 'Hence (C)', 'E. D', 'Answer: (B)',
                           'no option', '', '[A]', ' ( A ) '):
                with self.subTest(count=count, answer=answer):
                    self.assertEqual(flow.extract_characters_regex(answer, count),
                                     scope['extract_answer'](answer, dataset) or '')
        flow.qa_model = SimpleNamespace(question_answering=lambda *a, **kw: 'Final answer: D')
        result = flow.video_close_qa('which?', ['a', 'b', 'c', 'd'], 'D')
        self.assertEqual(result['pred_choice'], 'D')
        self.assertEqual(result['acc'], 1.)
        self.assertEqual(flow.extract_characters_regex('(G)', 7), 'G')

    def test_rvs_prefix_replay_offline_questions_and_csv_contract(self):
        class Runtime:
            config = InfiniPotConfig()
            def __init__(self):
                self.seen = []
            def reset(self):
                self.video = None
            def set_video(self, video):
                self.video = video
            def get_prompt(self, q, mc=False):
                return q
            def question_answering(self, inputs, **kwargs):
                self.seen.append((self.video, inputs['prompt']))
                return '(B)'
        sample = dict(video_id='v', video_path='unused', conversations=[
            dict(question='q', choices=['a', 'b', 'c', 'd'], answer='b', end_time=2),
            dict(question='q2', answer='yes', end_time=2),
            dict(question='q3', answer='no', end_time=4)])
        for cls in (InfiniPotStreamVQA, InfiniPotOfflineVQA):
            runtime = Runtime()
            with tempfile.TemporaryDirectory() as tmp:
                flow = cls(anno=[sample], save_dir=tmp, sample_fps=2, qa_model=runtime, num_chunks=1, chunk_idx=0)
                with patch('video_qa.infinipot_vqa.load_video', side_effect=lambda p, **kw: kw['end_time']) as load:
                    flow.analyze()
                self.assertEqual(load.call_count, 2 if cls.streaming else 1)
                self.assertEqual([v for v, _ in runtime.seen], [2, 2, 4] if cls.streaming else [None] * 3)
                self.assertEqual(flow.records[0]['qa_acc'], 100)
                self.assertIn('Respond with only the letter (A, B, C or D)', runtime.seen[0][1])
                self.assertNotIn('Best option:', runtime.seen[0][1])
                with (Path(tmp) / '1_0.csv').open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(len(rows), 3)
                self.assertEqual(rows[0]['correct_choice'], 'B')
                self.assertIsNone(runtime.video)


if __name__ == '__main__':
    unittest.main()
