"""Offline/streaming output and evaluation launcher failures without model weights."""
import csv
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from video_qa import base, run_eval, suprkv_offline_vqa, suprkv_stream_vqa
from video_qa.suprkv_stream_vqa import make_config
from model.suprkv.runtime import StreamingRuntime


class FakeRuntime:
    instances = []
    get_prompt = StreamingRuntime.get_prompt
    question_answering = StreamingRuntime.question_answering

    def __init__(self, config):
        self.config = config
        self.events = []
        self.frames = []
        self.startup = {'model_instances': 1, 'execution_mode': 'sequential'}
        self.instances.append(self)

    def close(self):
        self.events.append(('close',))

    def reset(self):
        self.events.append(('reset',))
        self.frames = []

    def append_frame(self, image, timestamp):
        self.frames.append(timestamp)
        self.events.append(('frame', timestamp))
        return self.memory_stats()

    def ask(self, question, cutoff=None, *, prompt=None, max_new_tokens=None):
        if cutoff is not None:
            assert all(timestamp <= cutoff for timestamp in self.frames)
        self.events.append(('ask', question, cutoff, tuple(self.frames), prompt, max_new_tokens))
        return {'answer': '(B)\nchosen'}

    def close_segment(self):
        self.events.append(('commit',))
        return self.memory_stats()

    def memory_stats(self):
        return {'frames': len(self.frames)}


def frames(path, fps):
    yield from ((timestamp, 'image') for timestamp in (0., 2., 4., 6.))


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
        default_annotation = self.directory / 'data/rvs/ego/ego4d_oe.json'
        default_annotation.parent.mkdir(parents=True)
        default_annotation.write_text(json.dumps(self.samples))
        previous_directory = Path.cwd()
        os.chdir(self.directory)
        self.addCleanup(os.chdir, previous_directory)
        FakeRuntime.instances = []

    def worker_argv(self, *extra):
        return ['--anno_path', str(self.annotation), '--save_dir', str(self.directory),
                '--sample_fps', '0.5', *extra]

    def worker_args(self, *extra):
        return base.build_parser('suprkv').parse_args(self.worker_argv(*extra))

    def launcher_args(self, *extra):
        return run_eval.build_parser().parse_args([
            '--method', 'suprkv', '--dataset', 'rvs_ego', *extra])

    def run_worker(self, *extra):
        with patch('model.suprkv.runtime.StreamingRuntime', FakeRuntime), \
                patch.object(suprkv_stream_vqa, 'sampled_frames', frames):
            base.work(suprkv_stream_vqa.SuPrKVStreamVQA, method='suprkv',
                      argv=self.worker_argv(*extra))

    def analyzer(self, runtime, sample):
        return suprkv_stream_vqa.SuPrKVStreamVQA(
            anno=[sample], save_dir=str(self.directory), sample_fps=.5, qa_model=runtime)

    def read_results(self, filename='1_0.csv'):
        with (self.directory / filename).open(newline='') as result:
            return list(csv.DictReader(result))

    def test_full_dataset_all_questions_model_loaded_once_per_worker(self):
        self.run_worker()
        rows = self.read_results()
        self.assertEqual(len(rows), 15)
        self.assertEqual([row['video_id'] for row in rows], [str(i) for i in range(5) for _ in range(3)])
        self.assertEqual(rows[0]['question'], 'first\nquestion')
        self.assertEqual(len(FakeRuntime.instances), 1)
        self.assertEqual(FakeRuntime.instances[0].events.count(('reset',)), 5)
        self.assertEqual(len((self.directory / '1_0.stats.jsonl').read_text().splitlines()), 5)

    def test_global_limit_before_partition_no_duplicates(self):
        for index in range(2):
            self.run_worker('--max_samples', '3', '--num_chunks', '2', '--chunk_idx', str(index))
        rows = self.read_results('2_0.csv') + self.read_results('2_1.csv')
        self.assertEqual([row['video_id'] for row in rows], [str(i) for i in range(3) for _ in range(3)])
        self.assertEqual(len(FakeRuntime.instances), 2)

    def test_streaming_uses_rekv_array_boundary_and_reuses_same_cutoff(self):
        runtime = FakeRuntime(make_config(self.worker_args('--device', 'cpu')))
        with patch.object(suprkv_stream_vqa, 'sampled_frames', frames):
            self.analyzer(runtime, self.samples[0]).analyze_a_video(self.samples[0])
        self.assertEqual([event[3] for event in runtime.events if event[0] == 'ask'],
                         [(0.,), (0.,), (0., 2.)])
        self.assertTrue(all(event[2] is None for event in runtime.events if event[0] == 'ask'))
        self.assertTrue(all(event[4:] == (None, None) for event in runtime.events if event[0] == 'ask'))
        self.assertNotIn(('frame', 4.), runtime.events)
        self.assertNotIn(('frame', 6.), runtime.events)

    def test_stream_worker_writes_rekv_fields_and_separate_method_stats(self):
        self.run_worker('--max_samples', '2')
        with (self.directory / '1_0.csv').open(newline='') as output:
            reader = csv.DictReader(output)
            self.assertEqual(reader.fieldnames, ['video_id', 'question', 'answer', 'pred_answer',
                                                 'retrieve_size', 'chunk_size'])
            rows = list(reader)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row['pred_answer'] == '(B)chosen' for row in rows))
        self.assertTrue(all(row['retrieve_size'] == '64' and row['chunk_size'] == '1' for row in rows))
        stats = [json.loads(line) for line in (self.directory / '1_0.stats.jsonl').read_text().splitlines()]
        self.assertEqual([entry['video_id'] for entry in stats], ['0', '1'])
        self.assertTrue(all(entry['config']['budget_tokens'] == 8192 for entry in stats))
        self.assertTrue(all(entry['memory'] == {'frames': 2} for entry in stats))
        self.assertFalse((self.directory / 'results.csv').exists())

    def test_offline_datasets_dispatch_to_evaluator(self):
        for dataset in ('mlvu', 'qaego4d', 'egoschema', 'activitynet_qa', 'cgbench'):
            with self.subTest(dataset=dataset), \
                    patch.object(run_eval, f'eval_{dataset}') as evaluate:
                run_eval.main(['--method', 'suprkv', '--dataset', dataset])
                evaluate.assert_called_once()

    def test_offline_worker_encodes_full_video_once_for_all_question_types(self):
        conversations = [
            dict(question='open question', answer='description', question_type='open'),
            dict(question='which one?', choices=['first', 'second'], answer='second',
                 question_type='action', end_time=0),
            dict(question='hidden answer', choices=['first', 'second'], answer=None),
        ]
        samples = [dict(video_id=str(i), video_path=f'{i}.mp4', conversations=conversations)
                   for i in range(2)]
        self.annotation.write_text(json.dumps(samples))
        with patch('model.suprkv.runtime.StreamingRuntime', FakeRuntime), \
                patch.object(suprkv_offline_vqa, 'sampled_frames', frames):
            base.work(suprkv_offline_vqa.SuPrKVOfflineVQA, method='suprkv',
                      argv=self.worker_argv('--max_new_tokens', '32'))
        self.assertEqual(len(FakeRuntime.instances), 1)
        runtime = FakeRuntime.instances[0]
        self.assertEqual(runtime.events.count(('reset',)), 2)
        self.assertEqual(runtime.events.count(('commit',)), 2)
        self.assertEqual(sum(event[0] == 'frame' for event in runtime.events), 8)
        asks = [event for event in runtime.events if event[0] == 'ask']
        self.assertEqual(len(asks), 6)
        self.assertTrue(all(event[3] == (0., 2., 4., 6.) for event in asks))
        self.assertEqual(asks[1][1], 'which one?')
        self.assertEqual(asks[1][4],
                         '\nQuestion: which one?\nOptions:\n(A) first\n(B) second\n'
                         'Only give the best option.<|im_end|><|im_start|>assistant\nBest option: (')
        self.assertEqual(asks[0][4], '\nopen question<|im_end|><|im_start|>assistant\n')
        self.assertEqual([event[5] for event in asks], [1024, 16, 16] * 2)
        for index, event in enumerate(runtime.events):
            if event[0] == 'ask':
                self.assertIn(runtime.events[index - 1][0], ('commit', 'ask'))
        self.assertEqual(runtime.events[-1], ('close',))
        rows = self.read_results()
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]['task'], 'open')
        self.assertEqual(rows[0]['pred_answer'], '(B)chosen')
        self.assertEqual(rows[1]['correct_choice'], 'B')
        self.assertEqual(rows[1]['pred_choice'], 'B')
        self.assertEqual(float(rows[1]['qa_acc']), 100.)
        self.assertEqual(rows[1]['task'], 'action')
        self.assertEqual(rows[2]['answer'], 'first')
        self.assertEqual(rows[2]['correct_choice'], 'A')
        self.assertEqual(float(rows[2]['qa_acc']), 0.)
        self.assertIsNone(samples[0]['conversations'][2]['answer'])
        stats = [json.loads(line) for line in (self.directory / '1_0.stats.jsonl').read_text().splitlines()]
        self.assertEqual([entry['video_id'] for entry in stats], ['0', '1'])
        self.assertTrue(all(entry['memory'] == {'frames': 4} for entry in stats))
        self.assertTrue(all(entry['config']['max_new_tokens'] == 32 for entry in stats))

    def test_offline_closes_frame_iterator_on_encoding_error_and_rejects_empty_video(self):
        for empty in (False, True):
            with self.subTest(empty=empty):
                closed = []

                def source(path, fps):
                    try:
                        if not empty:
                            yield 0., 'image'
                    finally:
                        closed.append(True)

                runtime = FakeRuntime(make_config(self.worker_args('--device', 'cpu')))
                runtime.append_frame = Mock(side_effect=ValueError('encoding failed'))
                analyzer = suprkv_offline_vqa.SuPrKVOfflineVQA(
                    anno=self.samples, save_dir=str(self.directory), sample_fps=.5, qa_model=runtime)
                with patch.object(suprkv_offline_vqa, 'sampled_frames', source), \
                        self.assertRaisesRegex(ValueError, 'no sampled frames' if empty else 'encoding failed'):
                    analyzer.analyze_a_video(self.samples[0])
                self.assertEqual(closed, [True])
                self.assertEqual(analyzer.record[(64, 1)], [])
                self.assertEqual(analyzer.video_stats, [])

    def test_offline_empty_choice_response_is_incorrect(self):
        runtime = FakeRuntime(make_config(self.worker_args('--device', 'cpu')))
        runtime.ask = Mock(return_value={'answer': '  \n'})
        analyzer = suprkv_offline_vqa.SuPrKVOfflineVQA(
            anno=self.samples, save_dir=str(self.directory), sample_fps=.5, qa_model=runtime)
        result = analyzer.video_close_qa('question', ['one', 'two'], 'A')
        self.assertEqual(result['pred_choice'], '')
        self.assertEqual(result['acc'], 0.)

    def test_offline_launchers_forward_suprkv_config_and_use_dataset_scorers(self):
        for dataset, scorer in (
                ('mlvu', 'eval_multiple_choice.py'), ('qaego4d', 'eval_multiple_choice.py'),
                ('egoschema', 'eval_egoschema.py'), ('activitynet_qa', 'eval_open_ended.py'),
                ('cgbench', 'eval_multiple_choice.py')):
            args = self.launcher_args('--dataset', dataset, '--budget_tokens', '4096',
                                      '--max_new_tokens', '16', '--packed_attention')
            with self.subTest(dataset=dataset), \
                    patch.object(run_eval, 'get_eval_settings', return_value=(1, str(self.directory))), \
                    patch.object(run_eval.multiprocessing, 'Process', return_value=Mock(exitcode=0)) as launch, \
                    patch.object(run_eval, 'exec') as execute:
                getattr(run_eval, f'eval_{dataset}')(args)
                command, sub, device = launch.call_args.kwargs['args']
                self.assertEqual(command[:3], ['python', '-m', 'video_qa.suprkv_offline_vqa'])
                self.assertTrue(sub)
                self.assertEqual(device, '0')
                self.assertNotIn('--n_local', command)
                worker_args = base.build_parser('suprkv').parse_args(command[3:])
                self.assertEqual(suprkv_stream_vqa.config_overrides(worker_args),
                                 suprkv_stream_vqa.config_overrides(args))
                self.assertTrue(worker_args.packed_attention)
                self.assertIn(scorer, execute.call_args.args[0])

    def test_invalid_timestamps_do_not_encode_frames_and_close_model(self):
        for cutoff in (1, -1, float('nan'), float('inf')):
            with self.subTest(cutoff=cutoff):
                self.samples[0]['conversations'][1]['end_time'] = cutoff
                self.annotation.write_text(json.dumps(self.samples))
                with self.assertRaisesRegex(ValueError, 'chronological'):
                    self.run_worker()
                self.assertEqual(FakeRuntime.instances[-1].events, [('close',)])

    def test_debug_and_repeated_runs_write_only_current_stats(self):
        self.run_worker()
        self.run_worker('--debug', 'true')
        with (self.directory / '1_0.csv').open(newline='') as output:
            self.assertEqual(len(list(csv.DictReader(output))), 3)
        self.assertEqual(len((self.directory / '1_0.stats.jsonl').read_text().splitlines()), 1)

    def test_launcher_forwards_parameters_and_uses_rekv_merge_commands(self):
        args = self.launcher_args('--max_samples', '2', '--num_chunks', '8',
                                  '--budget-tokens', '8192', '--packed-attention', '--sample_fps', '0.5',
                                  '--model-path', '/custom/model', '--recent-frames', '3',
                                  '--group-size', '64', '--ema-alpha', '0.3',
                                  '--min-segment-frames', '2', '--max-segment-frames', '16',
                                  '--summary-max-new-tokens', '128', '--max-new-tokens', '64')
        process = Mock(exitcode=0)
        with patch.object(run_eval.multiprocessing, 'Process', return_value=process) as launch, \
                patch.object(run_eval, 'exec') as execute:
            run_eval.eval_rvs_ego(args)
        _, directory = run_eval.get_eval_settings(args, args.dataset, 'data/rvs/ego/ego4d_oe.json')
        self.assertEqual(launch.call_count, 2)
        for index, call in enumerate(launch.call_args_list):
            command, sub, device = call.kwargs['args']
            self.assertIs(call.kwargs['target'], execute)
            self.assertTrue(sub)
            self.assertEqual(device, str(index))
            self.assertEqual(command[command.index('--num_chunks') + 1], '2')
            self.assertEqual(command[command.index('--max_samples') + 1], '2')
            self.assertEqual(command[command.index('--sample_fps') + 1], '0.5')
            self.assertEqual(command[command.index('--anno_path') + 1], 'data/rvs/ego/ego4d_oe.json')
            self.assertEqual(command[command.index('--save_dir') + 1], directory)
            self.assertIn('--packed_attention', command)
            worker_args = base.build_parser('suprkv').parse_args(command[3:])
            self.assertEqual(suprkv_stream_vqa.config_overrides(worker_args),
                             suprkv_stream_vqa.config_overrides(args))
        commands = [call.args[0] for call in execute.call_args_list]
        self.assertIn(f'rm -rf {directory}/tmp', commands)
        self.assertIn(f'head -n 1 {directory}/2_0.csv > {directory}/results.csv', commands)
        for index in range(2):
            self.assertIn(f'tail -n +2 {directory}/2_{index}.csv >> {directory}/results.csv', commands)
            self.assertIn(f'rm {directory}/2_{index}.csv', commands)
        self.assertIn(f'--pred_path {directory}/results.csv', commands[-1])

    def test_worker_failure_never_merges_or_scores_partial_results(self):
        for dataset in ('rvs_ego', 'rvs_movie', 'mlvu', 'qaego4d', 'egoschema', 'activitynet_qa', 'cgbench'):
            with self.subTest(dataset=dataset), \
                    patch.object(run_eval, 'get_eval_settings', return_value=(1, str(self.directory))), \
                    patch.object(run_eval.multiprocessing, 'Process', return_value=Mock(exitcode=1)), \
                    patch.object(run_eval, 'exec') as execute:
                with self.assertRaisesRegex(RuntimeError, 'workers failed'):
                    getattr(run_eval, f'eval_{dataset}')(self.launcher_args('--dataset', dataset))
                execute.assert_not_called()

    def test_merge_preserves_multiline_rows_and_clears_old_scores(self):
        for dataset in ('rvs_ego', 'rvs_movie'):
            with self.subTest(dataset=dataset):
                for index in range(2):
                    self.run_worker('--max_samples', '3', '--num_chunks', '2', '--chunk_idx', str(index))
                cache = self.directory / 'tmp'
                cache.mkdir()
                (cache / 'old.json').write_text('{}')
                original_exec = run_eval.exec
                def execute(command):
                    if not command.startswith('python '):
                        original_exec(command)
                with patch.object(run_eval, 'get_eval_settings', return_value=(2, str(self.directory))), \
                        patch.object(run_eval.multiprocessing, 'Process', return_value=Mock(exitcode=0)), \
                        patch.object(run_eval, 'exec', side_effect=execute) as commands:
                    getattr(run_eval, f'eval_{dataset}')(self.launcher_args('--dataset', dataset))
                rows = self.read_results('results.csv')
                self.assertEqual(len(rows), 9)
                self.assertEqual(rows[0]['question'], 'first\nquestion')
                self.assertFalse(cache.exists())
                self.assertFalse((self.directory / '2_0.csv').exists())
                self.assertFalse((self.directory / '2_1.csv').exists())
                self.assertIn('eval_open_ended.py', commands.call_args.args[0])

    def test_only_eval_uses_existing_results_and_correct_scorer(self):
        for dataset in ('rvs_ego', 'rvs_movie'):
            with self.subTest(dataset=dataset), \
                    patch.object(run_eval.multiprocessing, 'Process') as launch, \
                    patch.object(run_eval, 'exec') as execute, \
                    patch.object(run_eval, 'load_samples', side_effect=AssertionError('no annotation needed')):
                args = self.launcher_args('--dataset', dataset, '--only_eval')
                getattr(run_eval, f'eval_{dataset}')(args)
                launch.assert_not_called()
                execute.assert_called_once()
                self.assertIn('video_qa/eval/eval_open_ended.py', execute.call_args.args[0])

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
            with self.subTest(extra=extra), patch.object(run_eval, 'eval_rvs_ego') as launch, \
                    redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run_eval.main(['--method', 'suprkv', '--dataset', 'rvs_ego', *extra])
                launch.assert_not_called()

    def test_shared_model_resolution_and_actual_runtime_defaults(self):
        from model.suprkv.runtime import StreamingConfig
        config = make_config(self.worker_args('--device', 'cpu'))
        self.assertEqual(config, StreamingConfig(device='cpu'))
        _, directory = run_eval.get_eval_settings(self.launcher_args(), 'rvs_ego', self.annotation)
        self.assertTrue(directory.endswith(f'/budget_{config.budget_tokens}'))
        config = make_config(self.worker_args('--device', 'cpu', '--model', 'llava_ov_0.5b',
                                             '--retrieve_size', '32', '--budget-tokens', '8192'))
        self.assertIn('0.5b', config.model_path)
        self.assertEqual(config.topk, 32)
        self.assertEqual(config.budget_tokens, 8192)
        config = make_config(self.worker_args('--device', 'cpu', '--model_path', '/custom/model',
                                             '--max_new_tokens', '128'))
        self.assertEqual(config.model_path, '/custom/model')
        self.assertEqual(config.max_new_tokens, 128)


if __name__ == '__main__':
    unittest.main()
