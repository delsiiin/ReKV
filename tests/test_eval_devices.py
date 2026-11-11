"""GPU assignment and loader contracts without downloading weights or using CUDA."""
from contextlib import ExitStack
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from model.loading import load_pretrained
from model.video_backbone import model_family
from video_qa import run_eval
from video_qa.methods import METHODS, validate_arguments
from video_qa.model_paths import MODEL_PATHS


class EvalDeviceTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_selection_priority_order_and_uuid_support(self):
        args = SimpleNamespace(gpus=None)
        self.assertEqual(run_eval.worker_gpus(args, 2), ['0', '1'])
        os.environ['CUDA_VISIBLE_DEVICES'] = '5, 2,7'
        self.assertEqual(run_eval.worker_gpus(args, 2), ['5', '2'])
        args.gpus = 'GPU-abc-123,MIG-GPU-def-456/1/0'
        self.assertEqual(run_eval.worker_gpus(args, 2), ['GPU-abc-123', 'MIG-GPU-def-456/1/0'])
        self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '5, 2,7')

    def test_invalid_selection_fails_before_any_worker_starts(self):
        for flags in (['--gpus', '2'], ['--gpus', ''], ['--gpus', '2,,5'],
                      ['--gpus', '-1'], ['--gpus', 'cuda:0'],
                      ['--device', 'cuda:1'], ['--device', 'auto'], ['--device', 'cpu']):
            args = run_eval.build_parser().parse_args([
                '--method', 'native', '--dataset', 'mlvu', '--num_chunks', '2', *flags])
            with self.subTest(flags=flags), \
                    patch.object(run_eval, 'load_samples', return_value=[{}, {}]), \
                    patch.object(run_eval.multiprocessing, 'Process') as launch, \
                    self.assertRaises(ValueError):
                run_eval.evaluate(args, 'mlvu')
            launch.assert_not_called()
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            with self.assertRaises(ValueError):
                run_eval.worker_gpus(SimpleNamespace(gpus=None), 1)

    def test_every_method_and_model_assigns_one_selected_gpu_per_worker(self):
        for method_name, method in METHODS.items():
            for model in MODEL_PATHS:
                allowed = {'streamkv': ('onevision',), 'infinipot': ('qwen2_5_vl', 'onevision')}.get(method_name)
                if allowed and model_family(model) not in allowed:
                    continue
                args = run_eval.build_parser().parse_args([
                    '--method', method_name, '--model', model, '--dataset', 'mlvu',
                    '--num_chunks', '2', '--gpus', '5,2'])
                validate_arguments(args, method)
                with self.subTest(method=method_name, model=model), \
                        patch.object(run_eval, 'load_samples', return_value=[{}, {}]), \
                        patch.object(run_eval.multiprocessing, 'Process', return_value=Mock(exitcode=0)) as launch, \
                        patch.object(run_eval, 'exec'):
                    run_eval.evaluate(args, 'mlvu')
                self.assertEqual([call.kwargs['args'][2] for call in launch.call_args_list], ['5', '2'])
                self.assertEqual(launch.return_value.start.call_count, 2)

    def test_subprocess_receives_one_gpu_without_changing_parent_environment(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='5,2', TEST_EVAL_ENV='kept'), \
                patch.object(run_eval.subprocess, 'run') as execute:
            run_eval.exec(['python', '-m', 'worker'], sub=True, device='2')
            self.assertEqual(execute.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '2')
            self.assertEqual(execute.call_args.kwargs['env']['TEST_EVAL_ENV'], 'kept')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '5,2')

    def test_only_eval_requires_no_gpu_configuration(self):
        args = run_eval.build_parser().parse_args(['--dataset', 'mlvu', '--only_eval', '--gpus', ''])
        with patch.object(run_eval.multiprocessing, 'Process') as launch, \
                patch.object(run_eval, 'exec') as execute:
            run_eval.evaluate(args, 'mlvu')
        launch.assert_not_called()
        execute.assert_called_once()


class SingleDeviceLoadingTests(unittest.TestCase):
    def test_shared_loader_preserves_explicit_single_device_and_rejects_auto(self):
        model_class = Mock()
        for device, expected in ((None, 'cuda:0'), ('cuda', 'cuda:0'),
                                 ('cuda:2', 'cuda:2'), ('cpu', 'cpu')):
            load_pretrained(model_class, 'local', device=device, dtype=torch.float16)
            self.assertEqual(model_class.from_pretrained.call_args.kwargs['device_map'], expected)
        with self.assertRaises((ValueError, RuntimeError)):
            load_pretrained(model_class, 'local', device='auto')

    def test_all_backends_reach_transformers_with_single_gpu_and_original_precision(self):
        class StopBeforeWeights(Exception):
            pass

        backbones = {
            'onevision': ('LlavaOnevisionForConditionalGeneration', 'LlavaOnevisionProcessor',
                          'llava_onevision_rekv.LlavaOneVision_ReKV'),
            'video_llava': ('VideoLlavaForConditionalGeneration', 'VideoLlavaProcessor',
                            'video_llava_rekv.VideoLlava_ReKV'),
            'qwen2_5_vl': ('Qwen2_5_VLForConditionalGeneration', 'Qwen2_5_VLProcessor',
                          'qwen2_5_vl_rekv.Qwen2_5_VL_ReKV'),
        }
        for method_name, method in METHODS.items():
            for family, (model_class, processor_class, rekv_class) in backbones.items():
                allowed = {'streamkv': ('onevision',), 'infinipot': ('qwen2_5_vl', 'onevision')}.get(method_name)
                if allowed and family not in allowed:
                    continue
                preset = {'onevision': 'llava_ov_7b', 'video_llava': 'video_llava_7b',
                          'qwen2_5_vl': 'qwen2_5_vl_7b'}[family]
                args = run_eval.build_parser().parse_args([
                    '--method', method_name, '--model', preset, '--dataset', 'mlvu'])
                validate_arguments(args, method)
                target = ('model.rekv.' + rekv_class if method_name == 'rekv'
                          else 'transformers.' + model_class)
                with self.subTest(method=method_name, family=family), ExitStack() as stack:
                    processor = stack.enter_context(patch('transformers.' + processor_class + '.from_pretrained'))
                    processor.return_value.tokenizer.return_value.input_ids = torch.tensor([[1, 2]])
                    stack.enter_context(patch('torch.cuda.set_device'))
                    loader = stack.enter_context(patch(target + '.from_pretrained', side_effect=StopBeforeWeights))
                    with self.assertRaises(StopBeforeWeights):
                        method.load(args)
                    options = loader.call_args.kwargs
                    self.assertEqual(options['device_map'], 'cuda:0')
                    self.assertTrue(options['low_cpu_mem_usage'])
                    self.assertEqual(options['dtype'], 'auto' if method_name == 'infinipot' else torch.float16)
                    if method_name == 'infinipot':
                        self.assertEqual(options['attn_implementation'], 'flash_attention_2')
                    elif method_name == 'suprkv':
                        self.assertEqual(options['attn_implementation'], 'sdpa')


if __name__ == '__main__':
    unittest.main()
