"""Native Qwen video processing/generation and evaluation routes, no downloads."""
import copy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen2VLVideoProcessor,
                          Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration)

from model.native.runtime import NativeRuntime
from model.native.qwen2_5_vl import Qwen2_5_VLNativeRuntime
from model.video_backbone import INITIAL_PROMPTS
from video_qa import base, run_eval
from video_qa.methods import get_method, validate_arguments
from video_qa.model_paths import MODEL_PATHS
from model.video_backbone import model_family
from video_qa.native_offline_vqa import NativeOfflineVQA
from video_qa.native_stream_vqa import NativeStreamVQA
from test_qwen2_5_vl import tiny_config


def native_processor():
    # A real fast tokenizer and multimodal processor with small, local vocab.
    vocab = {f'token{i}': i for i in range(64)}
    for index, name in [(0, '[UNK]'), (1, '<|im_start|>'), (58, '<|vision_start|>'),
                        (59, '<|vision_end|>'), (60, '<|image_pad|>'),
                        (61, '<|video_pad|>'), (62, '[PAD]'), (63, '<|im_end|>')]:
        del vocab[f'token{index}']
        vocab[name] = index
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]', eos_token='<|im_end|>',
        additional_special_tokens=['<|im_start|>', '<|vision_start|>', '<|vision_end|>',
                                   '<|image_pad|>', '<|video_pad|>'])
    options = dict(patch_size=2, temporal_patch_size=2, merge_size=2,
                   size=dict(shortest_edge=64, longest_edge=256))
    return Qwen2_5_VLProcessor(tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(**options),
        video_processor=Qwen2VLVideoProcessor(**options))


class NativeQwenTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = Qwen2_5_VLForConditionalGeneration(tiny_config()).eval()
        model.generation_config.pad_token_id = 62
        self.runtime = Qwen2_5_VLNativeRuntime(model, native_processor(), 'qwen2_5_vl')
        self.frames = torch.randint(0, 256, (4, 16, 24, 3), dtype=torch.uint8)

    @torch.inference_mode()
    def test_native_processor_generation_parity_and_question_isolation(self):
        runtime = self.runtime
        for index in range(3):
            runtime.append_frame(self.frames[index], index * 2.)
        prompt = runtime.get_prompt('describe', mc=True)
        inputs = runtime._prepare_inputs(prompt)
        expected_inputs = runtime.processor(
            text=[INITIAL_PROMPTS['qwen2_5_vl'] + '<|video_pad|>' + prompt],
            videos=[self.frames[:3]], input_data_format='channels_last', do_sample_frames=False,
            video_metadata=[dict(total_num_frames=3, fps=.5, frames_indices=[0, 1, 2])],
            return_tensors='pt')
        self.assertEqual(inputs.keys(), expected_inputs.keys())
        for key in inputs:
            torch.testing.assert_close(inputs[key], expected_inputs[key])
        self.assertEqual(int(inputs.video_grid_thw[0, 0]), 2)  # Native temporal merging, odd padding.
        self.assertEqual(float(inputs.second_per_grid_ts[0]), 4.)
        self.assertNotEqual(int((inputs.mm_token_type_ids == 2).sum()), 3 * 196)
        reference = copy.deepcopy(runtime.model)
        expected_ids = reference.generate(**expected_inputs, max_new_tokens=4,
            do_sample=False, num_beams=1, use_cache=True, return_dict_in_generate=False)
        expected = runtime.tokenizer.decode(expected_ids[0, expected_inputs.input_ids.shape[1]:],
            skip_special_tokens=True, spaces_between_special_tokens=False, clean_up_tokenization_spaces=True)
        positions = []
        def record(module, args, kwargs):
            positions.append(kwargs['position_ids'].clone())
        handle = runtime.decoder.register_forward_pre_hook(record, with_kwargs=True)
        try:
            self.assertEqual(runtime.question_answering({'prompt': prompt}, max_new_tokens=4), expected)
        finally:
            handle.remove()
        self.assertFalse(torch.equal(positions[0][-2], positions[0][-1]))
        self.assertIsNone(runtime.model.model.rope_deltas)
        runtime.question_answering({'prompt': runtime.get_prompt('another question')}, max_new_tokens=2)
        self.assertEqual(runtime.question_answering({'prompt': prompt}, max_new_tokens=4), expected)
        for index in range(3):
            torch.testing.assert_close(runtime.frames[index], self.frames[index])
        runtime.append_frame(self.frames[3], 6.)
        # Padding the odd prefix must not survive when the true next frame arrives.
        actual = runtime._prepare_inputs(prompt)
        self.assertEqual(int(actual.video_grid_thw[0, 0]), 2)
        self.assertFalse(torch.equal(actual.pixel_values_videos, inputs.pixel_values_videos))
        runtime.question_answering({'prompt': prompt}, max_new_tokens=2)
        self.assertFalse(hasattr(runtime.decoder, '_old_forward'))
        self.assertFalse(any(hasattr(layer.self_attn, '_old_forward') for layer in runtime.decoder.layers))
        runtime.reset()
        self.assertFalse(runtime.frames or runtime.timestamps)
        text_only = runtime._prepare_inputs(prompt)
        self.assertNotIn('video_grid_thw', text_only)
        self.assertFalse(torch.isin(text_only.input_ids, torch.tensor([58, 59, 61])).any())
        runtime.question_answering({'prompt': prompt}, max_new_tokens=2)

    def test_limits_eos_and_cleanup_on_failure(self):
        runtime = self.runtime
        runtime.max_new_tokens = 3
        runtime.model.generation_config.eos_token_id = None
        prompt = {'prompt': runtime.get_prompt('describe')}
        with patch.object(runtime.model, 'generate', wraps=runtime.model.generate) as generate:
            runtime.question_answering(prompt, max_new_tokens=1)
        self.assertEqual(generate.call_args.kwargs['max_new_tokens'], 3)
        inputs = runtime._prepare_inputs(prompt['prompt'])
        with torch.inference_mode():
            first = int(runtime.model.generate(**inputs, max_new_tokens=1)[0, -1])
        runtime.model.generation_config.eos_token_id = first
        with patch.object(runtime.tokenizer, 'decode', wraps=runtime.tokenizer.decode) as decode:
            runtime.question_answering(prompt, max_new_tokens=9)
        self.assertEqual(decode.call_args.args[0].tolist(), [first])
        def fail(**kwargs):
            runtime.model.model.rope_deltas = torch.tensor([[100]])
            raise RuntimeError('generation failed')
        with patch.object(runtime.model, 'generate', side_effect=fail), self.assertRaisesRegex(RuntimeError, 'failed'):
            runtime.question_answering(prompt)
        self.assertIsNone(runtime.model.model.rope_deltas)
        runtime.close()
        self.assertFalse(runtime.frames or runtime.timestamps)

    def test_native_loader_presets_and_all_evaluation_routes(self):
        for name in (name for name in MODEL_PATHS if model_family(name) == 'qwen2_5_vl'):
            with patch.object(Qwen2_5_VLForConditionalGeneration, 'from_pretrained',
                              return_value=self.runtime.model) as loader, \
                    patch.object(Qwen2_5_VLProcessor, 'from_pretrained', return_value=self.runtime.processor):
                runtime = NativeRuntime.load(name, '/local/qwen', device='cpu', max_new_tokens=3, sample_fps=.5)
            self.assertIsInstance(runtime, Qwen2_5_VLNativeRuntime)
            self.assertEqual(loader.call_args.args, ('/local/qwen',))
            self.assertEqual(loader.call_args.kwargs['dtype'], torch.float32)
            self.assertEqual(runtime.max_new_tokens, 3)
            runtime.append_frame(self.frames[0], 0.)
            self.assertEqual(float(runtime._prepare_inputs(runtime.get_prompt('describe')).second_per_grid_ts[0]), 4.)
            for dataset in run_eval.DATASETS:
                args = run_eval.build_parser().parse_args([
                    '--method', 'native', '--model', name, '--dataset', dataset,
                    '--model_path', '/local/qwen', '--max_new_tokens', '3'])
                validate_arguments(args, get_method('native'))
                with patch.object(run_eval, 'load_samples', return_value=[{}]), \
                        patch.object(run_eval.multiprocessing, 'Process', return_value=Mock(exitcode=0)) as launch, \
                        patch.object(run_eval, 'exec'):
                    run_eval.evaluate(args, dataset)
                command, _, devices = launch.call_args.kwargs['args']
                self.assertEqual(command[2], f'video_qa.native_{run_eval.DATASETS[dataset][1]}_vqa')
                worker = base.build_parser('native').parse_args(command[3:])
                self.assertEqual(worker.model, name)
                self.assertEqual(worker.model_path, '/local/qwen')
                self.assertEqual(devices, '0,1,2,3' if name.endswith('72b') else '0')

    def test_offline_and_streaming_flows_use_only_visible_frames(self):
        runtime = self.runtime
        runtime.max_new_tokens = 1
        sample = dict(video_id='v', video_path='unused', conversations=[
            dict(question='describe', answer='answer', end_time=t) for t in [0, 2, 2, 6]])
        def frames(_sample):
            yield from ((i * 2., frame.numpy()) for i, frame in enumerate(self.frames))
        seen = []
        original = runtime._prepare_inputs
        def prepare(prompt):
            seen.append(tuple(runtime.timestamps))
            return original(prompt)
        with tempfile.TemporaryDirectory() as directory:
            for cls, expected in [(NativeStreamVQA, [(), (0.,), (0.,), (0., 2., 4.)]),
                                  (NativeOfflineVQA, [(0., 2., 4., 6.)] * 4)]:
                seen.clear()
                analyzer = cls(anno=[sample], save_dir=directory, sample_fps=.5, qa_model=runtime)
                with patch.object(analyzer, 'frame_source', side_effect=frames), \
                        patch.object(runtime, '_prepare_inputs', side_effect=prepare):
                    analyzer.analyze_a_video(sample)
                self.assertEqual(seen, expected)
                self.assertEqual(len(analyzer.records), 4)

    def test_native_module_does_not_import_retrieval_dependencies(self):
        subprocess.run([sys.executable, '-c', '''
import sys
from model.native.qwen2_5_vl import Qwen2_5_VLNativeRuntime
assert not any(name.startswith(('model.rekv', 'model.suprkv', 'flash_attn')) for name in sys.modules)
'''], check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
