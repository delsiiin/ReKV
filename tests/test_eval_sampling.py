"""CPU-only checks for global sample limits and evaluation worker scheduling."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from video_qa import run_eval
from video_qa.sampling import load_samples, nonnegative_int, split_samples


class EvalSamplingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'annotations.json'
        self.samples = [
            {'video_id': i, 'conversations': [{'question': 'one'}, {'question': 'two'}]}
            for i in range(10)
        ]
        self.path.write_text(json.dumps(self.samples))

    def test_limit_is_global_and_preserves_all_questions(self):
        selected = load_samples(self.path, 7)
        chunks = split_samples(selected, 3)
        self.assertEqual([sample for chunk in chunks for sample in chunk], self.samples[:7])
        self.assertEqual(sum(len(chunk) for chunk in chunks), 7)
        self.assertTrue(all(len(sample['conversations']) == 2 for sample in selected))

    def test_unlimited_and_limit_above_dataset_size(self):
        for limit in (0, 100):
            self.assertEqual(load_samples(self.path, limit), self.samples)

    def test_partition_has_no_missing_duplicate_or_empty_chunks(self):
        for size in range(1, 25):
            samples = list(range(size))
            for count in range(1, size + 1):
                chunks = split_samples(samples, count)
                self.assertEqual(len(chunks), count)
                self.assertEqual([sample for chunk in chunks for sample in chunk], samples)
                lengths = [len(chunk) for chunk in chunks]
                self.assertGreater(min(lengths), 0)
                self.assertLessEqual(max(lengths) - min(lengths), 1)

    def test_invalid_limits_and_chunks(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            nonnegative_int('-1')
        for count in (0, -1, 11):
            with self.assertRaises(ValueError):
                split_samples(self.samples, count)
        with self.assertRaises(ValueError):
            split_samples([], 1)

    def test_launcher_caps_workers_and_isolates_limited_results(self):
        args = SimpleNamespace(model='llava_ov_7b', retrieve_size=64, sample_fps=1,
                               max_samples=2, num_chunks=8, only_eval=False)
        count, directory = run_eval.get_eval_settings(args, 'qaego4d', self.path)
        self.assertEqual(count, 2)
        self.assertTrue(directory.endswith('/samples_2'))
        args.max_samples = 0
        count, directory = run_eval.get_eval_settings(args, 'qaego4d', self.path)
        self.assertEqual(count, 8)
        self.assertEqual(directory, 'results/llava_ov_7b/qaego4d/64-1')
        args.only_eval = True
        args.max_samples = 2
        with patch.object(run_eval, 'load_samples', side_effect=AssertionError('must not read annotations')):
            _, directory = run_eval.get_eval_settings(args, 'qaego4d', self.path)
        self.assertTrue(directory.endswith('/samples_2'))

    def test_all_dataset_routes_forward_global_limit_and_merge_actual_chunks(self):
        args = SimpleNamespace(model='llava_ov_7b', retrieve_size=64, sample_fps=1,
                               n_local=15000, debug='false', max_samples=2,
                               num_chunks=8, only_eval=False)
        datasets = ('mlvu', 'qaego4d', 'egoschema', 'activitynet_qa',
                    'rvs_ego', 'rvs_movie', 'cgbench')
        for dataset in datasets:
            with self.subTest(dataset=dataset), \
                 patch.object(run_eval, 'load_samples', return_value=self.samples[:2]), \
                 patch.object(run_eval.multiprocessing, 'Process') as process, \
                 patch.object(run_eval, 'exec') as execute:
                getattr(run_eval, f'eval_{dataset}')(args)
                self.assertEqual(process.call_count, 2)
                for index, call in enumerate(process.call_args_list):
                    command, _, device = call.kwargs['args']
                    self.assertEqual(command[command.index('--max_samples') + 1], '2')
                    self.assertEqual(command[command.index('--num_chunks') + 1], '2')
                    self.assertEqual(command[command.index('--chunk_idx') + 1], str(index))
                    self.assertEqual(device, str(index))
                    self.assertTrue(command[command.index('--save_dir') + 1].endswith('/samples_2'))
                commands = [call.args[0] for call in execute.call_args_list]
                merge_commands = [command for command in commands if command.startswith('tail ')]
                self.assertEqual(len(merge_commands), 2)
                self.assertIn('/2_0.csv', merge_commands[0])
                self.assertIn('/2_1.csv', merge_commands[1])


if __name__ == '__main__':
    unittest.main()
