"""Dataset coverage, temporal isolation and launcher failures without model weights."""
import csv
from contextlib import redirect_stderr
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from video_qa import run_eval, suprkv_stream_vqa
from video_qa.suprkv_stream_vqa import make_config


class FakeRuntime:
    instances = []

    def __init__(self, config):
        self.config = config
        self.events = []
        self.frames = []
        self.startup = {'model_instances': 1, 'execution_mode': 'sequential'}
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.events.append(('close',))

    def reset(self):
        self.events.append(('reset',))
        self.frames = []

    def append_frame(self, image, timestamp):
        self.frames.append(timestamp)
        self.events.append(('frame', timestamp))
        return self.memory_stats()

    def ask(self, question, cutoff=None):
        if cutoff is not None:
            assert all(timestamp <= cutoff for timestamp in self.frames)
        self.events.append(('ask', question, cutoff, tuple(self.frames)))
        return {'answer': '(B)\nchosen'}

    def close_segment(self):
        self.events.append(('commit',))
        return self.memory_stats()

    def memory_stats(self):
        return {'frames': len(self.frames)}


def frames(path, fps):
    return iter((timestamp, 'image') for timestamp in (0., 2., 4., 6.))


class SuPrKVEvalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.annotation = self.directory / 'annotations.json'
        self.samples = [dict(video_id=str(i), video_path=f'{i}.mp4', conversations=[
            dict(question='first\nquestion', answer='ground truth', end_time=2),
            dict(question='same time', answer='ground truth', end_time=2),
            dict(question='next time', answer='ground truth', end_time=5),
        ]) for i in range(5)]
        self.annotation.write_text(json.dumps(self.samples))
        FakeRuntime.instances = []

    def worker_args(self, *extra):
        return suprkv_stream_vqa.build_worker_parser().parse_args([
            '--dataset', 'rvs_ego', '--anno_path', str(self.annotation),
            '--save_dir', str(self.directory), '--sample_fps', '0.5', *extra])

    def launcher_args(self, *extra):
        return run_eval.build_parser().parse_args([
            '--method', 'suprkv', '--dataset', 'rvs_ego',
            '--anno_path', str(self.annotation), '--save_dir', str(self.directory),
            '--skip_eval', *extra])

    def run_worker(self, args):
        # Dataset orchestration tests also run in an environment without torch.
        with patch.object(suprkv_stream_vqa, 'make_config', return_value=SimpleNamespace(topk=64)):
            suprkv_stream_vqa.run_worker(args, FakeRuntime, frames)

    def read_results(self):
        with (self.directory / 'results.csv').open(newline='') as result:
            return list(csv.DictReader(result))

    def test_full_dataset_all_questions_model_loaded_once_per_worker(self):
        self.run_worker(self.worker_args())
        suprkv_stream_vqa.merge_results(self.directory, 1)
        rows = self.read_results()
        self.assertEqual(len(rows), 15)
        self.assertEqual([row['video_id'] for row in rows], [str(i) for i in range(5) for _ in range(3)])
        self.assertEqual(rows[0]['question'], 'first\nquestion')
        self.assertEqual(len(FakeRuntime.instances), 1)
        self.assertEqual(FakeRuntime.instances[0].events.count(('reset',)), 5)
        self.assertEqual(len((self.directory / '1_0.stats.jsonl').read_text().splitlines()), 5)

    def test_global_limit_before_partition_no_duplicates(self):
        for index in range(2):
            self.run_worker(self.worker_args('--max_samples', '3', '--num_chunks', '2', '--chunk_idx', str(index)))
        suprkv_stream_vqa.merge_results(self.directory, 2)
        rows = self.read_results()
        self.assertEqual([row['video_id'] for row in rows], [str(i) for i in range(3) for _ in range(3)])
        self.assertEqual(len(FakeRuntime.instances), 2)

    def test_streaming_uses_rekv_array_boundary_and_reuses_same_cutoff(self):
        runtime = FakeRuntime(None)
        suprkv_stream_vqa.evaluate_sample(runtime, self.samples[0], .5, 64, True, frames)
        self.assertEqual([event[3] for event in runtime.events if event[0] == 'ask'],
                         [(0.,), (0.,), (0., 2.)])
        self.assertTrue(all(event[2] is None for event in runtime.events if event[0] == 'ask'))
        self.assertNotIn(('frame', 4.), runtime.events)
        self.assertNotIn(('frame', 6.), runtime.events)

    def test_offline_encodes_full_video_and_preserves_choice_schema(self):
        question = dict(question='pick', answer='second', choices=['first', 'second'], question_type='plotQA')
        sample = dict(video_id='choice', video_path='choice.mp4', conversations=[question])
        runtime = FakeRuntime(None)
        rows = suprkv_stream_vqa.evaluate_sample(runtime, sample, .5, 64, False, frames)
        self.assertEqual(runtime.events[-1][3], (0., 2., 4., 6.))
        self.assertEqual(runtime.events[-2], ('commit',))
        self.assertIn('(B) second', runtime.events[-1][1])
        self.assertEqual(rows[0]['qa_acc'], 100.)
        self.assertEqual(rows[0]['correct_choice'], 'B')
        self.assertEqual(rows[0]['pred_choice'], 'B')
        self.assertEqual(rows[0]['task'], 'plotQA')
        question['answer'] = None
        row = suprkv_stream_vqa.answer_row(runtime, sample, question, 64)
        self.assertIsNone(row['answer'])
        self.assertEqual(row['qa_acc'], '')

    def test_every_dataset_worker_writes_scorer_compatible_rows(self):
        for dataset, (_, scorer) in suprkv_stream_vqa.DATASETS.items():
            with self.subTest(dataset=dataset):
                samples = json.loads(json.dumps(self.samples[:2]))
                if scorer != 'open_ended':
                    for sample in samples:
                        for question in sample['conversations']:
                            question['choices'] = ['incorrect', 'ground truth']
                self.annotation.write_text(json.dumps(samples))
                self.run_worker(self.worker_args('--dataset', dataset))
                suprkv_stream_vqa.merge_results(self.directory, 1)
                rows = self.read_results()
                self.assertEqual(len(rows), 6)
                if scorer != 'open_ended':
                    self.assertTrue(all(row['qa_acc'] == '100.0' for row in rows))

    def test_invalid_timestamps_fail_before_loading_model(self):
        self.samples[0]['conversations'][1]['end_time'] = 1
        self.annotation.write_text(json.dumps(self.samples))
        with self.assertRaisesRegex(ValueError, 'chronological'):
            self.run_worker(self.worker_args())
        self.assertEqual(FakeRuntime.instances, [])

    def test_launcher_respects_visible_devices_and_forwards_parameters(self):
        args = self.launcher_args('--max_samples', '2', '--num_chunks', '8',
                                  '--budget-tokens', '8192', '--packed-attention', '--sample_fps', '0.5')
        process = Mock()
        process.wait.return_value = 0
        process.poll.return_value = 0
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3,5'}), \
                patch.object(suprkv_stream_vqa.subprocess, 'Popen', return_value=process) as launch, \
                patch.object(suprkv_stream_vqa, 'merge_results') as merge:
            suprkv_stream_vqa.launch_evaluation(args)
        self.assertEqual(launch.call_count, 2)
        for index, call in enumerate(launch.call_args_list):
            command = call.args[0]
            self.assertEqual(command[command.index('--num_chunks') + 1], '2')
            self.assertEqual(command[command.index('--max_samples') + 1], '2')
            self.assertEqual(command[command.index('--budget_tokens') + 1], '8192')
            self.assertEqual(command[command.index('--sample_fps') + 1], '0.5')
            self.assertIn('--packed_attention', command)
            self.assertEqual(call.kwargs['env']['CUDA_VISIBLE_DEVICES'], ['3', '5'][index])
        merge.assert_called_once_with(str(self.directory), 2)

    def test_worker_failure_never_merges_or_scores_partial_results(self):
        process = Mock()
        process.wait.return_value = 1
        process.poll.return_value = 1
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), \
                patch.object(suprkv_stream_vqa.subprocess, 'Popen', return_value=process), \
                patch.object(suprkv_stream_vqa, 'merge_results') as merge, \
                patch.object(suprkv_stream_vqa.subprocess, 'run') as score:
            with self.assertRaisesRegex(RuntimeError, 'workers failed'):
                suprkv_stream_vqa.launch_evaluation(self.launcher_args())
        merge.assert_not_called()
        score.assert_not_called()

    def test_new_predictions_invalidate_old_open_ended_scores(self):
        cache = self.directory / 'tmp'
        cache.mkdir()
        (cache / 'old.json').write_text('{}')
        process = Mock()
        process.wait.return_value = 0
        process.poll.return_value = 0
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), \
                patch.object(suprkv_stream_vqa.subprocess, 'Popen', return_value=process), \
                patch.object(suprkv_stream_vqa, 'merge_results'):
            suprkv_stream_vqa.launch_evaluation(self.launcher_args())
        self.assertFalse(cache.exists())

    def test_insufficient_visible_devices_rejected_before_launch(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), \
                patch.object(suprkv_stream_vqa.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'visible GPUs'):
                suprkv_stream_vqa.launch_evaluation(self.launcher_args('--num_chunks', '2'))
        launch.assert_not_called()

    def test_only_eval_uses_existing_results_and_correct_scorer(self):
        for dataset, (_, scorer) in suprkv_stream_vqa.DATASETS.items():
            with self.subTest(dataset=dataset), \
                    patch.object(suprkv_stream_vqa.subprocess, 'Popen') as launch, \
                    patch.object(suprkv_stream_vqa.subprocess, 'run') as score, \
                    patch.object(run_eval, 'load_samples', side_effect=AssertionError('no annotation needed')):
                args = self.launcher_args('--dataset', dataset, '--only_eval')
                args.skip_eval = False
                suprkv_stream_vqa.launch_evaluation(args)
                launch.assert_not_called()
                self.assertIn(f'video_qa.eval.eval_{scorer}', score.call_args.args[0])
                self.assertTrue(score.call_args.kwargs['check'])

    def test_cli_spellings_and_separate_results(self):
        hyphenated = self.worker_args('--sample-fps', '.5', '--retrieve-size', '32', '--budget-tokens', '8192')
        underscored = self.worker_args('--sample_fps', '.5', '--retrieve_size', '32', '--budget_tokens', '8192')
        self.assertEqual(vars(hyphenated), vars(underscored))
        args = self.launcher_args('--budget_tokens', '8192')
        _, directory = run_eval.get_eval_settings(args, 'rvs_ego', self.annotation)
        self.assertTrue(directory.startswith('results/suprkv/'))
        self.assertTrue(directory.endswith('/budget_8192'))

    def test_rekv_specific_parameter_and_unsupported_model_are_rejected(self):
        for extra in (['--n_local', '15000'], ['--model', 'longva_7b']):
            with self.subTest(extra=extra), patch.object(suprkv_stream_vqa, 'launch_evaluation') as launch, \
                    redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run_eval.main(['--method', 'suprkv', '--dataset', 'rvs_ego', *extra])
                launch.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires the rekv environment')
    def test_shared_model_resolution_and_actual_runtime_defaults(self):
        from model.suprkv.runtime import StreamingConfig
        config = make_config(self.worker_args('--device', 'cpu'))
        self.assertEqual(config, StreamingConfig(device='cpu'))
        config = make_config(self.worker_args('--device', 'cpu', '--model', 'llava_ov_0.5b',
                                             '--retrieve_size', '32', '--budget-tokens', '8192'))
        self.assertIn('0.5b', config.model_path)
        self.assertEqual(config.topk, 32)
        self.assertEqual(config.budget_tokens, 8192)
        config = make_config(self.worker_args('--device', 'cpu', '--model_path', '/custom/model'),
                             max_new_tokens=256)
        self.assertEqual(config.model_path, '/custom/model')
        self.assertEqual(config.max_new_tokens, 256)


if __name__ == '__main__':
    unittest.main()
