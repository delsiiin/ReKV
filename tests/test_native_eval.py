"""Native full-context inference and method-neutral scheduling, without downloads."""
from contextlib import redirect_stderr
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from transformers import (
    Qwen2Config, LlamaConfig, LlavaOnevisionConfig, VideoLlavaConfig,
    LlavaOnevisionForConditionalGeneration, VideoLlavaForConditionalGeneration,
    LlavaOnevisionVideoProcessor, VideoLlavaVideoProcessor,
)

from model.native.runtime import NativeRuntime
from model.video_backbone import INITIAL_PROMPTS
from video_qa import base, run_eval
from video_qa.methods import METHODS
from video_qa.methods.common import Method
from video_qa.native_offline_vqa import NativeOfflineVQA
from video_qa.native_stream_vqa import NativeStreamVQA


class Tokenizer:
    eos_token_id = None

    def __init__(self, bos=False):
        self.bos = bos
        self.decoded = []

    def __call__(self, text, **kwargs):
        ids = ([1] if self.bos else []) + [3 + ord(char) % 27 for char in text]
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def decode(self, tokens, **kwargs):
        self.decoded = list(tokens)
        return ' '.join(map(str, tokens))


def tiny_runtime(family):
    onevision = family == 'onevision'
    text = (Qwen2Config if onevision else LlamaConfig)(
        vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    config = (LlavaOnevisionConfig if onevision else VideoLlavaConfig)(
        text_config=text,
        vision_config=dict(model_type='siglip_vision_model' if onevision else 'clip_vision_model',
                           hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                           num_attention_heads=4, image_size=8, patch_size=2),
        vision_feature_layer=-1, vision_feature_select_strategy='full')
    model = (LlavaOnevisionForConditionalGeneration if onevision else VideoLlavaForConditionalGeneration)(config).eval()
    video_processor = (LlavaOnevisionVideoProcessor if onevision else VideoLlavaVideoProcessor)(
        size={'shortest_edge': 8}, crop_size={'height': 8, 'width': 8})
    processor = SimpleNamespace(tokenizer=Tokenizer(bos=not onevision), video_processor=video_processor)
    return NativeRuntime(model, processor, family)


class NativeRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(24)
        torch.set_num_threads(1)

    @torch.inference_mode()
    def test_native_cached_generation_matches_full_recomputation_without_question_leakage(self):
        for family in ('onevision', 'video_llava'):
            with self.subTest(family=family):
                runtime = tiny_runtime(family)
                for index in range(3):
                    runtime.append_frame(np.full((8, 8, 3), index * 50, dtype=np.uint8), index)
                retained = [frame.clone() for frame in runtime.frames]
                prompt = runtime.get_prompt('describe')
                # Independent, uncached greedy reference over the complete video prefix.
                prefix_ids = runtime.tokenizer(INITIAL_PROMPTS[family]).input_ids
                question_ids = runtime.tokenizer(prompt).input_ids
                if family == 'video_llava':
                    question_ids = question_ids[:, 1:]
                embed = runtime.model.get_input_embeddings()
                inputs = torch.cat([embed(prefix_ids), *runtime.frames, embed(question_ids)], dim=1)
                expected = []
                for _ in range(4):
                    out = runtime.decoder(inputs_embeds=inputs, use_cache=False)
                    token = int(runtime.model.lm_head(out.last_hidden_state[:, -1:])[0, -1].argmax())
                    expected.append(token)
                    inputs = torch.cat([inputs, embed(torch.tensor([[token]]))], dim=1)
                answer = runtime.question_answering({'prompt': prompt}, max_new_tokens=4)
                self.assertEqual(runtime.tokenizer.decoded, expected)
                runtime.question_answering({'prompt': runtime.get_prompt('unrelated')}, max_new_tokens=2)
                self.assertEqual(runtime.question_answering({'prompt': prompt}, max_new_tokens=4), answer)
                for before, after in zip(retained, runtime.frames):
                    torch.testing.assert_close(before, after)
                self.assertFalse(hasattr(runtime.decoder.layers[0].self_attn, '_streaming_n_local'))
                runtime.reset()
                self.assertEqual(runtime.frames, [])
                # A cutoff of zero is a text-only prefix and still supports native QA.
                runtime.question_answering({'prompt': prompt}, max_new_tokens=1)
                self.assertEqual(len(runtime.tokenizer.decoded), 1)

    @torch.inference_mode()
    def test_eos_and_explicit_generation_limit(self):
        runtime = tiny_runtime('onevision')
        inputs = {'prompt': runtime.get_prompt('q')}
        runtime.question_answering(inputs, max_new_tokens=1)
        first_token = runtime.tokenizer.decoded[0]
        runtime.tokenizer.eos_token_id = first_token
        runtime.question_answering(inputs, max_new_tokens=8)
        self.assertEqual(runtime.tokenizer.decoded, [first_token])
        runtime.tokenizer.eos_token_id = None
        runtime.max_new_tokens = 3
        runtime.question_answering(inputs, max_new_tokens=1)
        self.assertEqual(len(runtime.tokenizer.decoded), 3)

    def test_loader_uses_original_hf_classes(self):
        for model_name, family, cls, processor_cls in (
            ('llava_ov_7b', 'onevision', 'LlavaOnevisionForConditionalGeneration', 'LlavaOnevisionProcessor'),
            ('video_llava_7b', 'video_llava', 'VideoLlavaForConditionalGeneration', 'VideoLlavaProcessor'),
        ):
            original = tiny_runtime(family)
            with self.subTest(model=model_name), \
                    patch(f'transformers.{cls}.from_pretrained', return_value=original.model) as loader, \
                    patch(f'transformers.{processor_cls}.from_pretrained', return_value=original.processor):
                runtime = NativeRuntime.load(model_name, '/local/weights', device='cpu')
                self.assertIs(runtime.model, original.model)
                self.assertEqual(loader.call_args.kwargs['device_map'], 'cpu')
                self.assertEqual(loader.call_args.kwargs['dtype'], torch.float32)

    @torch.inference_mode()
    def test_longva_original_backbone_features_and_decoding(self):
        from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor
        from longva.model import LlavaQwenConfig, LlavaQwenForCausalLM
        with tempfile.TemporaryDirectory() as directory:
            vision_path = Path(directory) / 'vision'
            CLIPVisionModel(CLIPVisionConfig(
                hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                num_attention_heads=4, image_size=8, patch_size=2)).save_pretrained(vision_path)
            CLIPImageProcessor(size={'shortest_edge': 8}, crop_size={'height': 8, 'width': 8}).save_pretrained(vision_path)
            model = LlavaQwenForCausalLM(LlavaQwenConfig(
                vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, mm_vision_tower=str(vision_path),
                mm_vision_select_layer=-1, mm_hidden_size=32, mm_projector_type='linear',
                mm_spatial_pool_mode='average', mm_spatial_pool_stride=2)).eval()
            processor = model.get_vision_tower().image_processor
            processor.tokenizer = Tokenizer()
            runtime = NativeRuntime(model, processor, 'longva')
            runtime.append_frame(np.zeros((8, 8, 3), dtype=np.uint8), 0.)
            self.assertEqual(runtime.frames[0].shape, (1, 4, 32))
            answer = runtime.question_answering({'prompt': runtime.get_prompt('q')}, max_new_tokens=3)
            self.assertEqual(len(answer.split()), 3)
            runtime.close()
            self.assertEqual(runtime.frames, [])


class RecordingRuntime:
    processor = None

    def __init__(self):
        self.frames, self.questions, self.limits = [], [], []
        self.closed = False

    def reset(self):
        self.frames = []

    def close(self):
        self.closed = True

    def append_frame(self, frame, timestamp):
        self.frames.append(int(frame[0, 0, 0]))

    def get_prompt(self, question, mc=False):
        return question

    def question_answering(self, inputs, max_new_tokens):
        self.questions.append(tuple(self.frames))
        self.limits.append(max_new_tokens)
        return 'B)\nanswer'


class NativeEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.video_path = self.directory / 'video.npy'
        np.save(self.video_path, np.arange(10, dtype=np.uint8)[:, None, None, None].repeat(3, axis=3))
        self.sample = dict(video_id='v', video_path=str(self.video_path), conversations=[
            dict(question='first', answer='answer', end_time=0),
            dict(question='second', answer='answer', end_time=2),
            dict(question='same', answer='answer', end_time=2),
            dict(question='last', answer='answer', end_time=100),
        ])

    def test_stream_boundary_reuses_visual_features_without_future_frames(self):
        runtime = RecordingRuntime()
        analyzer = NativeStreamVQA(anno=[self.sample], save_dir=str(self.directory),
                                   sample_fps=.5, qa_model=runtime)
        analyzer.analyze_a_video(self.sample)
        self.assertEqual(runtime.questions, [(), (0,), (0,), (0, 2, 4, 6, 9)])
        self.assertEqual(runtime.limits, [256] * 4)
        self.assertEqual(len(analyzer.records), 4)

    def test_worker_native_csv_all_question_types_and_cleanup(self):
        self.sample['conversations'] = [dict(question='open', answer='answer'),
            dict(question='choice', choices=['a', 'b'], answer='b', question_type='action')]
        annotation = self.directory / 'anno.json'
        annotation.write_text(json.dumps([self.sample]))
        runtime = RecordingRuntime()
        with patch.object(NativeRuntime, 'load', return_value=runtime):
            base.work(NativeOfflineVQA, method='native', argv=[
                '--anno_path', str(annotation), '--save_dir', str(self.directory), '--sample_fps', '.5'])
        self.assertEqual(runtime.questions, [(0, 2, 4, 6, 9)] * 2)
        self.assertEqual(runtime.limits, [1024, 16])
        self.assertTrue(runtime.closed)
        with (self.directory / '1_0.csv').open() as source:
            reader = csv.DictReader(source)
            rows = list(reader)
            self.assertNotIn('retrieve_size', reader.fieldnames)
            self.assertNotIn('chunk_size', reader.fieldnames)
        self.assertEqual(rows[1]['qa_acc'], '100.0')
        self.assertEqual(rows[1]['task'], 'action')
        self.assertEqual(rows[0]['pred_answer'], 'B)answer')

    def test_worker_releases_runtime_on_failure(self):
        annotation = self.directory / 'anno.json'
        self.sample['conversations'][0]['end_time'] = -1
        annotation.write_text(json.dumps([self.sample]))
        runtime = RecordingRuntime()
        with patch.object(NativeRuntime, 'load', return_value=runtime), self.assertRaisesRegex(ValueError, 'chronological'):
            base.work(NativeStreamVQA, method='native', argv=[
                '--anno_path', str(annotation), '--save_dir', str(self.directory)])
        self.assertTrue(runtime.closed)
        self.assertFalse(runtime.questions)

    def test_launcher_native_routing_and_foreign_options(self):
        for dataset in run_eval.DATASETS:
            args = run_eval.build_parser().parse_args([
                '--method', 'native', '--dataset', dataset, '--model', 'llava_ov_72b',
                '--model_path', '/custom/weights', '--max_new_tokens', '32'])
            with self.subTest(dataset=dataset), \
                    patch.object(run_eval, 'load_samples', return_value=[self.sample]), \
                    patch.object(run_eval.multiprocessing, 'Process', return_value=Mock(exitcode=0)) as launch, \
                    patch.object(run_eval, 'exec'):
                run_eval.evaluate(args, dataset)
                command, _, device = launch.call_args.kwargs['args']
                mode = run_eval.DATASETS[dataset][1]
                self.assertIn(f'video_qa.native_{mode}_vqa', command)
                worker_args = base.build_parser('native').parse_args(command[3:])
                self.assertEqual(worker_args.max_new_tokens, 32)
                self.assertEqual(worker_args.model_path, '/custom/weights')
                self.assertEqual(device, '0,1,2,3')
                self.assertNotIn('--retrieve_size', command)
                self.assertIn(f'results/native/llava_ov_72b/{dataset}/fps_1', command)
        for option in ('--retrieve_size', '--n_local', '--retrieve_chunk_size', '--budget_tokens'):
            with self.subTest(option=option), redirect_stderr(io.StringIO()), \
                    patch.object(run_eval, 'eval_mlvu') as evaluate, self.assertRaises(SystemExit):
                run_eval.main(['--method', 'native', '--dataset', 'mlvu', option, '1'])
            evaluate.assert_not_called()

    def test_only_eval_needs_no_annotation_and_paths_are_distinct(self):
        paths = []
        for method in ('native', 'rekv', 'suprkv'):
            args = run_eval.build_parser().parse_args(['--method', method, '--dataset', 'mlvu', '--only_eval'])
            with patch.object(run_eval, 'load_samples', side_effect=AssertionError('no input needed')):
                paths.append(run_eval.get_eval_settings(args, 'mlvu', 'missing')[1])
        self.assertEqual(len(set(paths)), 3)

    def test_new_method_registers_without_changes_to_shared_runner_or_worker(self):
        runtime = RecordingRuntime()

        class ExampleMethod(Method):
            name = 'example'
            options = ('example_option',)

            def add_arguments(self, parser):
                parser.add_argument('--example_option', type=int, default=None)

            def result_dir(self, args, dataset):
                return f'results/example/{dataset}'

            def load(self, args):
                return runtime, None

            def close(self, model):
                model.close()

            def analyzer_kwargs(self, args):
                return dict(record_metadata={'example_option': args.example_option})

        annotation = self.directory / 'anno.json'
        annotation.write_text(json.dumps([self.sample]))
        with patch.dict(METHODS, example=ExampleMethod()):
            args = run_eval.build_parser().parse_args([
                '--method', 'example', '--dataset', 'mlvu', '--example_option', '7'])
            with patch.object(run_eval, 'load_samples', return_value=[self.sample]), \
                    patch.object(run_eval.multiprocessing, 'Process', return_value=Mock(exitcode=0)) as launch, \
                    patch.object(run_eval, 'exec'):
                run_eval.evaluate(args, 'mlvu')
            command = launch.call_args.kwargs['args'][0]
            self.assertIn('video_qa.example_offline_vqa', command)
            self.assertIn('--example_option', command)
            # A compatible lifecycle can reuse a flow without modifying BaseVQA.
            base.work(NativeOfflineVQA, method='example', argv=[
                '--anno_path', str(annotation), '--save_dir', str(self.directory),
                '--example_option', '7'])
        self.assertTrue(runtime.closed)
        with (self.directory / '1_0.csv').open() as output:
            rows = list(csv.DictReader(output))
        self.assertTrue(all(row['example_option'] == '7' for row in rows))
        self.assertNotIn('retrieve_size', rows[0])

    def test_existing_local_scorers_accept_native_results_without_retrieval_columns(self):
        results = self.directory / 'results.csv'
        with results.open('w') as output:
            writer = csv.DictWriter(output, fieldnames=[
                'video_id', 'question', 'answer', 'pred_answer', 'pred_choice', 'correct_choice', 'qa_acc'])
            writer.writeheader()
            writer.writerow(dict(video_id='v', question='q', answer='b', pred_answer='B) answer',
                                 pred_choice='B', correct_choice='B', qa_acc=100))
        score = subprocess.run([sys.executable, 'video_qa/eval/eval_multiple_choice.py',
                                '--save_dir', str(self.directory)], check=True, capture_output=True, text=True)
        self.assertIn('Average qa_acc: 100.00', score.stdout)
        subprocess.run([sys.executable, 'video_qa/eval/eval_egoschema.py', '--save_dir', str(self.directory)],
                       check=True, capture_output=True, text=True)
        with (self.directory / 'submission.csv').open() as output:
            self.assertEqual(list(csv.DictReader(output)), [{'q_uid': 'v', 'answer': '1'}])

    def test_help_and_native_imports_do_not_load_optimization_dependencies(self):
        script = '''
import sys
from video_qa import run_eval, base, native_offline_vqa, native_stream_vqa
from model.native.runtime import NativeRuntime
run_eval.build_parser()
base.build_parser('native')
assert not any(name.startswith(('model.rekv', 'model.suprkv', 'flash_attn', 'longva')) for name in sys.modules)
'''
        subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
