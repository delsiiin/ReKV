"""CPU scheduling, prefix isolation, and CLI regressions for asynchronous ReKV."""
from contextlib import contextmanager, nullcontext
from dataclasses import FrozenInstanceError
import gc
import tempfile
import threading
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import Mock, patch

import numpy as np
import torch

from model.rekv.abstract_rekv import Abstract_ReKV
from model.rekv.async_runtime import single_cuda_device
from model.rekv.attention.kv_cache_manager import ContextManager, CudaCache, VectorTensor
from model.rekv.attention.snapshot import KVSnapshot, LayerSnapshot, SnapshotRetrieval
from video_qa import run_eval
from video_qa.base import work
from video_qa.base import BaseVQA
from video_qa.methods import get_method, validate_arguments
from video_qa.rekv_offline_vqa import ReKVOfflineVQA
from video_qa.rekv_stream_vqa import ReKVStreamVQA


class FakeModel:
    def __init__(self):
        self.frames = []
        self.questions = []
        self.clear_count = 0
        self.encoded_second = threading.Event()
        self.encoded_third = threading.Event()
        self.encoded_fifth = threading.Event()
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.fail_encode = False
        self.fail_answer = False

    def clear_cache(self):
        self.clear_count += 1
        self.frames.clear()

    def encode_init_prompt(self):
        pass

    def encode_video(self, frames):
        assert torch.is_inference_mode_enabled()
        if self.fail_encode:
            raise RuntimeError('encoding failed')
        self.frames.extend(frames[:, 0, 0, 0].tolist())
        if len(self.frames) == 2:
            self.encoded_second.set()
        if len(self.frames) == 3:
            self.encoded_third.set()
        if len(self.frames) == 5:
            self.encoded_fifth.set()

    def get_prompt(self, question, mc=False):
        return question

    def question_answering(self, inputs, max_new_tokens, kv_snapshot):
        assert torch.is_inference_mode_enabled()
        self.started.set()
        if not self.release.wait(5):
            raise TimeoutError('test did not release QA')
        if self.fail_answer:
            raise RuntimeError('answer failed')
        self.questions.append((inputs['question'], kv_snapshot, max_new_tokens, threading.get_ident()))
        return 'A\n'


class CPURuntime:
    def __init__(self, model):
        self.model = model
        self.created = 0
        self.live = 0
        self.peak = 0
        self.closed = False
        self.two_created = threading.Event()

    @contextmanager
    def encoding(self):
        with torch.inference_mode():
            yield

    def capture(self):
        self.created += 1
        self.live += 1
        self.peak = max(self.peak, self.live)
        if self.created == 2:
            self.two_created.set()
        return tuple(self.model.frames)

    def answer(self, callback, video, question, snapshot):
        try:
            with torch.inference_mode():
                return callback(video, question, max_new_tokens=256, kv_snapshot=snapshot)
        finally:
            self.live -= 1

    def close(self):
        self.closed = True


class AsyncFlowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def make(self, cutoffs, count=6, fps=1, pending_groups=2):
        sample = dict(video_id='video', video_path='unused', conversations=[
            dict(question=f'q{i}', answer='A', end_time=t) for i, t in enumerate(cutoffs)])
        model = FakeModel()
        analyzer = ReKVStreamVQA(anno=[sample], save_dir=self.directory.name,
                                sample_fps=fps, qa_model=model, rekv_async=True,
                                rekv_async_max_pending_groups=pending_groups)
        runtime = CPURuntime(model)
        analyzer.runtime_factory = Mock(return_value=runtime)
        closed = []

        def frames(_):
            try:
                for i in range(count):
                    yield float(i), np.full((1, 1, 3), i, dtype=np.uint8)
            finally:
                closed.append(True)
        analyzer.frame_source = frames
        return analyzer, sample, model, runtime, closed

    def test_real_worker_overlap_backpressure_order_and_cleanup(self):
        analyzer, sample, model, runtime, closed = self.make([1, 2, 3, 4])
        model.release.clear()
        errors = []

        def run():
            try:
                analyzer.analyze_a_video(sample)
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(model.started.wait(5))
            self.assertTrue(runtime.two_created.wait(5))
            self.assertTrue(model.encoded_third.wait(5))
            self.assertFalse(model.questions)  # Two published views do not block disjoint writes.
            self.assertEqual(model.frames, [0, 1, 2])
            self.assertEqual(runtime.created, 2)  # Third task cannot capture until slot frees.
        finally:
            model.release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual([q[1] for q in model.questions], [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 3)])
        self.assertEqual([r['question'] for r in analyzer.records], ['q0', 'q1', 'q2', 'q3'])
        self.assertEqual({q[2] for q in model.questions}, {256})
        self.assertEqual(len({q[3] for q in model.questions}), 1)
        self.assertNotEqual(model.questions[0][3], thread.ident)
        self.assertLessEqual(runtime.peak, 2)
        self.assertEqual(runtime.live, 0)
        self.assertTrue(runtime.closed)
        self.assertEqual(closed, [True])
        self.assertEqual(model.frames, [])
        self.assertEqual(model.clear_count, 2)

    def test_whole_question_groups_share_views_while_writer_advances(self):
        analyzer, sample, model, runtime, _ = self.make([1] * 9 + [3] * 3 + [5] * 2)
        model.release.clear()
        errors = []

        def run():
            try:
                analyzer.analyze_a_video(sample)
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(model.started.wait(5))
            # Even Q1 of nine is blocked, but the encoder reaches the third cutoff.
            self.assertTrue(model.encoded_fifth.wait(5))
            self.assertEqual(runtime.created, 2)
            self.assertEqual(model.frames, [0, 1, 2, 3, 4])
            self.assertEqual(model.questions, [])
        finally:
            model.release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(runtime.created, 3)
        self.assertLessEqual(runtime.peak, 2)
        self.assertEqual(runtime.live, 0)
        self.assertEqual([q[1] for q in model.questions], [(0,)] * 9 + [(0, 1, 2)] * 3 + [(0, 1, 2, 3, 4)] * 2)
        self.assertTrue(all(model.questions[i][1] is model.questions[0][1] for i in range(9)))
        self.assertEqual([r['question'] for r in analyzer.records], [f'q{i}' for i in range(14)])

    def test_unbounded_writer_finishes_all_submissions_with_first_answer_blocked(self):
        analyzer, sample, model, runtime, _ = self.make([1, 2, 3, 4, 5, 6], pending_groups=0)
        model.release.clear()
        producer_done = threading.Event()
        analyzer.finish_encoding = producer_done.set
        errors = []

        def run():
            try:
                analyzer.analyze_a_video(sample)
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(model.started.wait(5))
            self.assertTrue(producer_done.wait(5))
            self.assertEqual(model.frames, list(range(6)))
            self.assertEqual(runtime.created, 6)
            self.assertEqual(model.questions, [])
            self.assertTrue(thread.is_alive())  # only the final result drain waits
        finally:
            model.release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual([q[1] for q in model.questions], [tuple(range(i)) for i in range(1, 7)])
        self.assertEqual([r['question'] for r in analyzer.records], [f'q{i}' for i in range(6)])
        stats = analyzer.async_video_stats[0]
        self.assertEqual(stats['max_pending_groups'], 0)
        self.assertEqual(stats['peak_pending_groups'], 6)
        self.assertEqual(stats['submitted_groups'], 6)
        self.assertEqual(stats['backpressure_seconds'], 0.)
        self.assertEqual(runtime.live, 0)

    def test_configurable_one_group_limit(self):
        analyzer, sample, model, runtime, _ = self.make([1, 2, 3], pending_groups=1)
        model.release.clear()
        errors = []

        def run():
            try:
                analyzer.analyze_a_video(sample)
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(model.started.wait(5))
            self.assertTrue(model.encoded_second.wait(5))
            self.assertEqual(runtime.created, 1)
            self.assertFalse(model.encoded_third.is_set())
        finally:
            model.release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(runtime.peak, 1)
        self.assertEqual(analyzer.async_video_stats[0]['submitted_groups'], 3)

    def test_unbounded_answer_failure_is_propagated_and_resources_closed(self):
        analyzer, sample, model, runtime, closed = self.make([1, 2, 3, 4, 5], pending_groups=0)
        model.fail_answer = True
        with self.assertRaisesRegex(RuntimeError, 'answer failed'):
            analyzer.analyze_a_video(sample)
        self.assertTrue(runtime.closed)
        self.assertEqual(closed, [True])
        self.assertEqual(model.frames, [])
        self.assertFalse(any(t.name.startswith('rekv-answer') for t in threading.enumerate()))

    def test_negative_group_limit_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nonnegative integer'):
            self.make([1], pending_groups=-1)

    def test_boundaries_repeated_cutoff_early_eof_and_multiple_videos(self):
        analyzer, sample, model, runtime, closed = self.make([0, 2, 2, 5, 100], count=3, fps=.5)
        expected = [(), (0,), (0,), (0, 1), (0, 1, 2)]
        for _ in range(2):
            analyzer.analyze_a_video(sample)
        self.assertEqual([q[1] for q in model.questions], expected * 2)
        self.assertEqual(len(closed), 2)
        self.assertEqual(runtime.live, 0)

    def test_choice_format_and_token_limit(self):
        analyzer, sample, model, _, _ = self.make([1])
        sample['conversations'][0].update(answer='first', choices=['first', 'second'])
        analyzer.analyze_a_video(sample)
        self.assertEqual(model.questions[0][2], 16)
        self.assertEqual(analyzer.records[0]['pred_choice'], 'A')
        self.assertEqual(analyzer.records[0]['qa_acc'], 100.)

    def test_no_questions_and_invalid_times_do_not_start_runtime(self):
        for cutoffs in ([], [-1], [float('nan')], [float('inf')], [2, 1]):
            analyzer, sample, _, _, _ = self.make(cutoffs)
            if cutoffs:
                with self.assertRaises(ValueError):
                    analyzer.analyze_a_video(sample)
            else:
                analyzer.analyze_a_video(sample)
            analyzer.runtime_factory.assert_not_called()

    def test_errors_join_worker_close_source_and_clear_cache(self):
        for failure in ('encode', 'answer'):
            analyzer, sample, model, runtime, closed = self.make([1, 2, 3])
            setattr(model, f'fail_{failure}', True)
            with self.assertRaisesRegex(RuntimeError, f'{"encoding" if failure == "encode" else "answer"} failed'):
                analyzer.analyze_a_video(sample)
            self.assertTrue(runtime.closed)
            self.assertEqual(closed, [True])
            self.assertEqual(model.frames, [])
            self.assertFalse(any(t.name.startswith('rekv-answer') for t in threading.enumerate()))

    def test_sync_also_releases_video_state_before_return_or_error(self):
        for failure in (None, 'encode', 'answer'):
            with self.subTest(failure=failure):
                analyzer, sample, model, _, closed = self.make([1, 2])
                analyzer.rekv_async = False
                model.question_answering = Mock(return_value='A')
                if failure == 'encode':
                    model.fail_encode = True
                elif failure == 'answer':
                    model.question_answering.side_effect = RuntimeError('answer failed')
                if failure:
                    with self.assertRaises(RuntimeError):
                        analyzer.analyze_a_video(sample)
                else:
                    analyzer.analyze_a_video(sample)
                    self.assertEqual(len(analyzer.records), 2)
                self.assertEqual(model.frames, [])
                self.assertEqual(closed, [True])
                analyzer.runtime_factory.assert_not_called()

    def test_initialization_and_source_close_failures_still_clear_model(self):
        analyzer, sample, model, runtime, _ = self.make([1])
        model.encode_init_prompt = Mock(side_effect=RuntimeError('initialization failed'))
        with self.assertRaisesRegex(RuntimeError, 'initialization failed'):
            analyzer.analyze_a_video(sample)
        self.assertTrue(runtime.closed)
        self.assertEqual(model.clear_count, 2)

        analyzer, sample, model, runtime, _ = self.make([1])
        frames = Mock()
        frames.__iter__ = Mock(return_value=iter([(0., np.zeros((1, 1, 3), dtype=np.uint8))]))
        frames.close.side_effect = RuntimeError('source close failed')
        analyzer.frame_source = lambda _: frames
        with self.assertRaisesRegex(RuntimeError, 'source close failed'):
            analyzer.analyze_a_video(sample)
        self.assertTrue(runtime.closed)
        self.assertEqual(model.clear_count, 2)
        self.assertEqual(model.frames, [])


def cache_fixture(frames, *, offloaded=False, topk=2, chunk_size=1, device='cpu'):
    """Distinct frame keys make the expected ranking explicit (newest first)."""
    cache = ContextManager(None, n_init=2, n_local=64, block_size=2,
                           max_cached_block=8, topk=topk, chunk_size=chunk_size, exc_block_size=2)
    init = torch.full((1, 1, 2, 4), -1., device=device)
    video = torch.arange(1, frames + 1, device=device, dtype=torch.float32).repeat_interleave(2)
    video = video[None, None, :, None].expand(1, 1, -1, 4).contiguous()
    cache.local_k = torch.cat((init, video), dim=2)
    cache.init_exc = offloaded
    cache.init_k, cache.init_v = (init.clone(), init.clone()) if offloaded else (init[:, :, :0], init[:, :, :0])
    cache.global_remainder = ((video[:, :, :0], video[:, :, :0]) if offloaded
                              else (cache.local_k.clone(), cache.local_k.clone()))
    vectors = VectorTensor(8, torch.float32, torch.device(device))
    if offloaded and frames:
        vectors.append(torch.arange(1, frames + 1, device=device, dtype=torch.float32)[:, None].expand(-1, 8).contiguous())
    cache.block_k = [vectors]
    cache.global_blocks = [[SimpleNamespace(cpu_data=tuple(
        video[0, :, 2 * i:2 * i + 2].cpu().clone() for _ in range(2))) for i in range(frames)]] if offloaded else [[]]
    cache.num_global_block = frames if offloaded else 0
    cache.length = 2 + frames * 2
    return cache


class SnapshotTests(unittest.TestCase):
    def test_ready_and_lifetime_registration_once_per_consumer_stream(self):
        layer = Mock(remainder=(torch.empty(0),))
        snapshot = KVSnapshot((layer,), Mock(), None)
        first, second = Mock(), Mock()
        with patch('torch.cuda.current_stream', return_value=first):
            snapshot.wait()
            snapshot.wait()
        with patch('torch.cuda.current_stream', return_value=second):
            snapshot.wait()
            snapshot.wait()
        first.wait_event.assert_called_once_with(snapshot.ready)
        second.wait_event.assert_called_once_with(snapshot.ready)
        self.assertEqual(layer.record_stream.call_count, 2)
        self.assertEqual([c.args[0] for c in layer.record_stream.call_args_list], [first, second])

    def test_failed_registration_can_be_retried(self):
        layer = Mock(remainder=(torch.empty(0),))
        layer.record_stream.side_effect = [RuntimeError('registration'), None]
        snapshot = KVSnapshot((layer,), Mock(), None)
        with patch('torch.cuda.current_stream', return_value=Mock()):
            with self.assertRaisesRegex(RuntimeError, 'registration'):
                snapshot.wait()
            snapshot.wait()
        self.assertEqual(layer.record_stream.call_count, 2)

    def test_pool_allocates_on_first_load_and_reuses_freed_slots(self):
        pool = CudaCache(2, 8, torch.float32, device='cpu')
        self.assertIsNone(pool.data)
        first, index = pool.alloc()
        first.fill_(7)
        storage = pool.data
        second, other = pool.alloc()
        self.assertNotEqual(index, other)
        pool.delete(index)
        reused, reused_index = pool.alloc()
        self.assertEqual(reused_index, index)
        self.assertIs(pool.data, storage)
        torch.testing.assert_close(reused, torch.full((8,), 7.))

    def test_video_reset_only_trims_allocator_when_requested(self):
        model = Abstract_ReKV(None, 2, [1, 2], 8, 2, 1)
        model.kv_cache = (torch.empty(1),)
        with patch('torch.cuda.empty_cache') as empty, patch('torch.cuda.ipc_collect') as ipc:
            model.clear_cache()
            self.assertIsNone(model.kv_cache)
            empty.assert_not_called()
            ipc.assert_not_called()
            model.clear_cache(release_allocator=True)
            empty.assert_called_once()
            ipc.assert_called_once()

    def test_selection_and_payload_match_known_frames_before_and_after_offload(self):
        query = torch.ones(1, 2, 3, 4)
        for count, topk, chunk, expected in ((0, 2, 1, []), (1, 2, 1, [0]),
                                           (5, 2, 1, [3, 4]), (5, 4, 2, [2, 3, 4])):
            for offloaded in (False, True):
                with self.subTest(count=count, offloaded=offloaded, chunk=chunk):
                    cache = cache_fixture(count, offloaded=offloaded, topk=topk, chunk_size=chunk)
                    snapshot = LayerSnapshot.capture(cache)
                    self.assertEqual(snapshot.select(query), [expected])
                    self.assertEqual(cache._calc_block_topk(query), [expected])
                    retrieved = SnapshotRetrieval(snapshot).get_retrieved_kv(query)
                    values = [-1., -1.] + [float(i + 1) for i in expected for _ in range(2)]
                    for tensor in retrieved:
                        torch.testing.assert_close(tensor[0, 0, :, 0], torch.tensor(values))
                    self.assertFalse(cache.to_retrieve)

    def test_shared_view_survives_append_offload_and_vector_growth(self):
        for offloaded in (False, True):
            cache = cache_fixture(5, offloaded=offloaded)
            snapshot = LayerSnapshot.capture(cache)
            before = snapshot.retrieve([[3, 4]])
            if offloaded:
                self.assertIs(snapshot.blocks[0][0][0], cache.global_blocks[0][0].cpu_data[0])
                old_ptr = snapshot.block_keys[0].untyped_storage().data_ptr()
                cache.block_k[0].append(torch.full((2, 8), 100.))  # same allocation, disjoint tail
                self.assertEqual(old_ptr, cache.block_k[0].data.untyped_storage().data_ptr())
                self.assertEqual(snapshot.select(torch.ones(1, 2, 1, 4)), [[3, 4]])
                cache.block_k[0].append(torch.zeros(40, 8))  # grows and replaces storage
                self.assertNotEqual(old_ptr, cache.block_k[0].data.untyped_storage().data_ptr())
                cache.global_blocks[0].append(SimpleNamespace(cpu_data=(torch.zeros(1),) * 2))
                self.assertEqual(len(snapshot.blocks[0]), 5)
                with self.assertRaises(IndexError):
                    snapshot.blocks[0][5]
            else:
                # The writer uses cat/replacement, never overwrites published data.
                cache.global_remainder = tuple(torch.cat((t, torch.full_like(t, 100.)), dim=2)
                                                for t in cache.global_remainder)
                cache.init_exc = True
            cache.init_k = torch.full_like(cache.init_k, 100.)
            cache.num_global_block = 99
            for actual, expected in zip(snapshot.retrieve([[3, 4]]), before):
                torch.testing.assert_close(actual, expected)
            self.assertEqual(snapshot.select(torch.ones(1, 2, 1, 4)), [[3, 4]])
            with self.assertRaises(FrozenInstanceError):
                snapshot.length = 100

    def test_capture_shares_tensor_storage_without_cloning_or_walking_history(self):
        class NoWalkList(list):
            def __iter__(self):
                raise AssertionError('Publication must not walk all historical blocks')

        for offloaded in (False, True):
            cache = cache_fixture(5, offloaded=offloaded)
            cache.global_blocks[0] = NoWalkList(cache.global_blocks[0])
            with patch.object(torch.Tensor, 'clone', side_effect=AssertionError('No payload copies')):
                snapshot = LayerSnapshot.capture(cache)
            self.assertIs(snapshot.init_kv[0], cache.init_k)
            self.assertIs(snapshot.remainder, cache.global_remainder)
            self.assertEqual(snapshot.block_keys[0].untyped_storage().data_ptr(),
                             cache.block_k[0].data.untyped_storage().data_ptr())
            if offloaded:
                self.assertIs(snapshot.blocks[0][-1], cache.global_blocks[0][4].cpu_data)
                self.assertEqual(len(snapshot.blocks[0][:]), 5)

    def test_reader_pins_old_payloads_until_release_after_encoder_reset(self):
        cache = cache_fixture(5, offloaded=True)
        payload = weakref.ref(cache.global_blocks[0][0].cpu_data[0])
        snapshot = LayerSnapshot.capture(cache)
        cache.global_blocks = [[]]  # reset must replace the append-only store
        gc.collect()
        self.assertIsNotNone(payload())
        self.assertEqual(snapshot.retrieve([[0]])[0][0, 0, 2, 0], 1.)
        del snapshot
        gc.collect()
        self.assertIsNone(payload())

    def test_private_results_external_indices_and_out_of_prefix(self):
        snapshot = LayerSnapshot.capture(cache_fixture(5, offloaded=True))
        first = snapshot.retrieve([[0, 3]])
        first[0].zero_()
        second = SnapshotRetrieval(snapshot, torch.tensor([[0, 3]])).get_retrieved_kv(None)
        self.assertNotEqual(first[0].data_ptr(), second[0].data_ptr())
        self.assertEqual(second[0][0, 0, 2, 0], 1.)
        for indices in ([[5]], [[-1]], [[0, 1, 2]]):
            with self.assertRaises(ValueError):
                snapshot.retrieve(indices)

    def test_sync_retrieval_exception_restores_flags(self):
        cache = cache_fixture(1)
        model = SimpleNamespace(kv_cache=(cache,), language_model=Mock(side_effect=RuntimeError('forward')))
        with self.assertRaisesRegex(RuntimeError, 'forward'):
            Abstract_ReKV._retrieve_question(model, torch.ones(1, 1))
        self.assertFalse(cache.to_retrieve)
        self.assertIsNone(cache.retrieved_block_indices)


class AsyncArgumentsTests(unittest.TestCase):
    def test_pending_group_options_validation_forwarding_and_result_isolation(self):
        method = get_method('rekv')
        for limit, suffix in ((0, 'async_unbounded'), (1, 'async_groups_1'), (2, 'async'), (4, 'async_groups_4')):
            args = run_eval.build_parser().parse_args([
                '--dataset', 'rvs_ego', '--rekv_async', '--rekv_async_max_pending_groups', str(limit), '--only_eval'])
            validate_arguments(args, method)
            command = method.worker_arguments(args)
            self.assertEqual(command[command.index('--rekv_async_max_pending_groups') + 1], str(limit))
            self.assertEqual(method.analyzer_kwargs(args)['rekv_async_max_pending_groups'], limit)
            self.assertTrue(run_eval.get_eval_settings(args, args.dataset, 'unused')[1].endswith('/' + suffix))
        for extra in ([], ['--rekv_async', '--method', 'native'], ['--rekv_async', '--method', 'suprkv']):
            args = run_eval.build_parser().parse_args([
                '--dataset', 'rvs_ego', '--rekv_async_max_pending_groups', '0', *extra])
            with self.assertRaises(ValueError):
                validate_arguments(args, get_method(args.method))
        args = run_eval.build_parser().parse_args([
            '--dataset', 'rvs_ego', '--rekv_async', '--rekv_async_max_pending_groups', '-1'])
        with self.assertRaises(ValueError):
            validate_arguments(args, method)

    def test_forwarding_paths_and_synchronous_default(self):
        for dataset in ('rvs_ego', 'rvs_movie'):
            args = run_eval.build_parser().parse_args(['--dataset', dataset, '--rekv_async', '--max_samples', '2'])
            method = get_method('rekv')
            validate_arguments(args, method)
            self.assertIn('--rekv_async', method.worker_arguments(args))
            with patch.object(run_eval, 'load_samples', return_value=[{}, {}]), \
                    patch.object(run_eval.multiprocessing, 'Process', return_value=Mock(exitcode=0)) as launch, \
                    patch.object(run_eval, 'exec'):
                run_eval.evaluate(args, dataset)
            command = launch.call_args.kwargs['args'][0]
            self.assertIn('--rekv_async', command)
            self.assertTrue(command[command.index('--save_dir') + 1].endswith('/samples_2/async'))
            args.only_eval = True
            with patch.object(run_eval, 'load_samples', side_effect=AssertionError):
                self.assertTrue(run_eval.get_eval_settings(args, dataset, 'unused')[1].endswith('/async'))
        args = run_eval.build_parser().parse_args(['--dataset', 'rvs_ego'])
        self.assertNotIn('--rekv_async', get_method('rekv').worker_arguments(args))
        self.assertNotIn('rekv_async', get_method('rekv').analyzer_kwargs(args))

    def test_unsupported_modes_rejected_before_load(self):
        for options in (['--dataset', 'mlvu'], ['--dataset', 'rvs_ego', '--model', 'llava_ov_72b'],
                        ['--dataset', 'rvs_ego', '--method', 'native'],
                        ['--dataset', 'rvs_ego', '--method', 'suprkv']):
            args = run_eval.build_parser().parse_args(options + ['--rekv_async'])
            with self.assertRaises(ValueError):
                validate_arguments(args, get_method(args.method))
        with patch.object(get_method('rekv'), 'load') as load, self.assertRaises(SystemExit):
            work(ReKVOfflineVQA, argv=['--save_dir', '/tmp/unused', '--anno_path', 'unused', '--rekv_async'])
        load.assert_not_called()

    def test_direct_worker_appends_async_directory_once(self):
        for limit, name in ((2, 'async'), (0, 'async_unbounded'), (1, 'async_groups_1')):
            for suffix in ('', '/' + name):
                with tempfile.TemporaryDirectory() as directory:
                    with patch('video_qa.base.load_samples', return_value=[{}]), \
                            patch.object(get_method('rekv'), 'load', return_value=(Mock(), None)), \
                            patch.object(ReKVStreamVQA, '__init__', return_value=None) as init, \
                            patch.object(ReKVStreamVQA, 'analyze'):
                        work(ReKVStreamVQA, argv=['--save_dir', directory + suffix,
                                                 '--anno_path', 'unused', '--rekv_async',
                                                 '--rekv_async_max_pending_groups', str(limit)])
                    self.assertEqual(init.call_args.kwargs['save_dir'], directory + '/' + name)
                    self.assertEqual(init.call_args.kwargs['rekv_async_max_pending_groups'], limit)

    def test_device_and_offload_validation(self):
        def model(devices, mapping=None, hook=None):
            item = SimpleNamespace(training=False, hf_device_map=mapping, _hf_hook=hook)
            item.modules = lambda: iter([item])
            item.parameters = lambda: iter([SimpleNamespace(device=torch.device(d)) for d in devices])
            return item
        self.assertEqual(single_cuda_device(model(['cuda:0'], {'': 0})), torch.device('cuda:0'))
        for candidate in (model(['cpu']), model(['cuda:0', 'cuda:1']),
                          model(['cuda:0'], {'layer': 'cpu'}), model(['cuda:0'], {'layer': 'disk'}),
                          model(['cuda:0'], {'layer': 1}),
                          model(['cuda:0'], hook=SimpleNamespace(hooks=[SimpleNamespace(offload=True)]))):
            with self.assertRaises(ValueError):
                single_cuda_device(candidate)


class GroupBenchmarkTests(unittest.TestCase):
    def test_latency_includes_wait_for_each_question_in_shared_view(self):
        from tests.benchmark_rekv_async import MeasuredRuntime, MeasuredVQA
        from model.rekv.async_runtime import ReKVAsyncRuntime

        snapshot = object()
        submitted = {id(snapshot): 5.}
        analyzer = MeasuredVQA.__new__(MeasuredVQA)
        analyzer.submitted, analyzer.qa_times = submitted, []
        analyzer.qa_model = SimpleNamespace(device='cuda:0')
        runtime = MeasuredRuntime.__new__(MeasuredRuntime)
        runtime.submitted = submitted
        questions = [{'question': 'a'}, {'question': 'b'}]

        def execute(callback, video, samples, view):
            return callback(video, samples, kv_snapshot=view)

        with patch.object(ReKVAsyncRuntime, 'answer', side_effect=execute), \
                patch.object(BaseVQA, 'answer_record', return_value={}), \
                patch('torch.cuda.nvtx.range', side_effect=lambda _: nullcontext()), \
                patch('torch.cuda.current_stream', return_value=Mock()), \
                patch('tests.benchmark_rekv_async.time.perf_counter', side_effect=[10., 11., 12., 13.]):
            result = runtime.answer(analyzer.answer_group, {'video_id': 'v'}, questions, snapshot)
        self.assertEqual(len(result), 2)
        self.assertEqual([q['response_seconds'] for q in analyzer.qa_times], [6., 8.])
        self.assertEqual([q['snapshot_and_queue_seconds'] for q in analyzer.qa_times], [5., 7.])
        self.assertEqual(submitted, {})


if __name__ == '__main__':
    unittest.main()
