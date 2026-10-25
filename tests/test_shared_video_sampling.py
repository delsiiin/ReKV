"""Compare the actual video inputs delivered to ReKV and SuPr-KV."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from observation.data import sample_indices
from video_qa.rekv_offline_vqa import ReKVOfflineVQA
from video_qa.rekv_stream_vqa import ReKVStreamVQA
from video_qa.sampling import sampled_frames
from video_qa.suprkv_stream_vqa import SuPrKVStreamVQA
from types import SimpleNamespace


class RecordingModel:
    def __init__(self):
        self.config = SimpleNamespace()
        self.frames = []
        self.questions = []
        self.batches = []

    def clear_cache(self):
        self.frames = []
        self.last_timestamp = float('-inf')

    reset = clear_cache

    def encode_init_prompt(self):
        pass

    def encode_video(self, frames):
        indices = frames[:, 0, 0, 0].tolist()
        self.batches.append(indices)
        self.frames.extend(indices)

    def get_prompt(self, question):
        return question

    def question_answering(self, inputs, **kwargs):
        self.questions.append(tuple(self.frames))
        return 'answer'

    def append_frame(self, image, timestamp):
        self.frames.append(int(image[0, 0, 0]))
        self.last_timestamp = timestamp

    def close_segment(self):
        pass

    def memory_stats(self):
        return {}

    def ask(self, question, cutoff=None):
        if cutoff is not None and cutoff < self.last_timestamp:
            raise ValueError('Question cutoff precedes encoded frames')
        self.questions.append(tuple(self.frames))
        return {'answer': 'answer'}


class VideoReaderFixture:
    def __init__(self, rate, count):
        self.rate, self.count = rate, count

    def get_avg_fps(self):
        return self.rate

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        class Frame:
            def asnumpy(self):
                return np.full((1, 1, 3), index, dtype=np.uint16)
        return Frame()


class SharedVideoSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'video.npy'
        self.video = np.arange(10, dtype=np.uint8)[:, None, None, None].repeat(3, axis=3)
        np.save(self.path, self.video)

    def sample(self, path, cutoffs):
        return dict(video_id='video', video_path=str(path), conversations=[
            dict(question=f'q{i}', answer='answer', start_time=0, end_time=cutoff)
            for i, cutoff in enumerate(cutoffs)])

    def compare_inputs(self, sample, fps, streaming):
        rekv, suprkv = RecordingModel(), RecordingModel()
        analyzer = (ReKVStreamVQA if streaming else ReKVOfflineVQA)(
            anno=[sample], save_dir=self.temporary.name, sample_fps=fps, qa_model=rekv)
        analyzer.analyze_a_video(sample)
        if streaming:
            candidate = SuPrKVStreamVQA(anno=[sample], save_dir=self.temporary.name,
                                       sample_fps=fps, qa_model=suprkv)
            candidate.analyze_a_video(sample)
            self.assertEqual(rekv.questions, suprkv.questions)
        else:
            expected = tuple(int(frame[0, 0, 0]) for _, frame in sampled_frames(sample['video_path'], fps))
            self.assertEqual(rekv.questions, [expected] * len(sample['conversations']))
        self.assertEqual(len(analyzer.record[(64, 1)]), len(sample['conversations']))
        return rekv

    def test_npy_original_rekv_linspace_including_odd_length_and_low_rate(self):
        for count, fps, expected in ((10, .5, [0, 2, 4, 6, 9]),
                                     (5, .5, [0, 4]), (5, .2, [0])):
            with self.subTest(count=count, fps=fps):
                np.save(self.path, self.video[:count])
                sampled = list(sampled_frames(self.path, fps))
                self.assertEqual([time for time, _ in sampled], expected)
                observation_indices, _ = sample_indices(count, 1., fps, 'npy')
                self.assertEqual(observation_indices, expected)
                for streaming in (True, False):
                    model = self.compare_inputs(self.sample(self.path, [100, 100]), fps, streaming)
                    self.assertEqual(model.questions, [tuple(expected)] * 2)

    def test_streaming_original_array_slices_without_duplicate_frames(self):
        model = self.compare_inputs(self.sample(self.path, [0, 2, 2, 5, 6, 100, 100]), .5, True)
        self.assertEqual(model.questions, [(), (0,), (0,), (0, 2), (0, 2, 4),
                                           (0, 2, 4, 6, 9), (0, 2, 4, 6, 9)])
        self.assertEqual(model.batches, [[0], [2], [4], [6], [9]])

    def test_rekv_encodes_each_frame_before_decoding_the_next(self):
        for streaming in (True, False):
            with self.subTest(streaming=streaming):
                model = RecordingModel()
                sample = self.sample('long.mp4', [130, 130, 150, 300])
                closed = []

                def source(path, fps):
                    try:
                        for index in range(150):
                            self.assertEqual(len(model.frames), index)
                            yield index, np.full((1, 1, 3), index, dtype=np.uint16)
                    finally:
                        closed.append(True)

                cls = ReKVStreamVQA if streaming else ReKVOfflineVQA
                module = 'rekv_stream_vqa' if streaming else 'rekv_offline_vqa'
                analyzer = cls(anno=[sample], save_dir=self.temporary.name,
                               sample_fps=1, qa_model=model)
                with patch(f'video_qa.{module}.sampled_frames', source), \
                        patch.object(analyzer, 'load_video', side_effect=AssertionError('eager loading')):
                    analyzer.analyze_a_video(sample)
                counts = (130, 130, 150, 150) if streaming else (150,) * 4
                self.assertEqual(model.questions, [tuple(range(n)) for n in counts])
                self.assertEqual(model.batches, [[index] for index in range(150)])
                self.assertEqual(closed, [True])

    def test_rekv_releases_decoder_after_encoding_or_decoding_failure(self):
        for streaming in (True, False):
            for fail_encoding in (True, False):
                with self.subTest(streaming=streaming, fail_encoding=fail_encoding):
                    model = RecordingModel()
                    sample = self.sample(self.path, [10])
                    closed = []

                    def source(path, fps):
                        try:
                            yield 0, self.video[0]
                            raise RuntimeError('decode failed')
                        finally:
                            closed.append(True)

                    cls = ReKVStreamVQA if streaming else ReKVOfflineVQA
                    module = 'rekv_stream_vqa' if streaming else 'rekv_offline_vqa'
                    analyzer = cls(anno=[sample], save_dir=self.temporary.name,
                                   sample_fps=1, qa_model=model)
                    if fail_encoding:
                        model.encode_video = Mock(side_effect=RuntimeError('encode failed'))
                    with patch(f'video_qa.{module}.sampled_frames', source):
                        with self.assertRaisesRegex(RuntimeError, 'encode failed' if fail_encoding else 'decode failed'):
                            analyzer.analyze_a_video(sample)
                    self.assertEqual(closed, [True])
                    self.assertEqual(model.questions, [])

    def test_offline_rekv_rejects_empty_video(self):
        np.save(self.path, self.video[:0])
        sample = self.sample(self.path, [1])
        model = RecordingModel()
        analyzer = ReKVOfflineVQA(anno=[sample], save_dir=self.temporary.name,
                                 sample_fps=1, qa_model=model)
        with self.assertRaisesRegex(ValueError, 'no sampled frames'):
            analyzer.analyze_a_video(sample)
        self.assertEqual(model.questions, [])

    def test_array_boundary_keeps_selected_frames_beyond_annotation_time(self):
        np.save(self.path, self.video[:5])
        # linspace selects [0, 4]; int(3.5 * .59) is 2, so both are visible.
        model = self.compare_inputs(self.sample(self.path, [1, 3.5, 4]), .59, True)
        self.assertEqual(model.questions, [(), (0, 4), (0, 4)])

    def test_mp4_original_rekv_integer_stride_matches_in_both_modes(self):
        for rate, fps, count, expected in (
                (25., 2., 51, [0, 12, 24, 36, 48]),
                (29.97, .5, 240, [0, 60, 120, 180]),
                (30., .5, 181, [0, 60, 120, 180]),
                (25., 25., 5, [0, 1, 2, 3, 4])):
            with self.subTest(rate=rate, fps=fps), \
                    patch('decord.VideoReader', return_value=VideoReaderFixture(rate, count)):
                sampled = list(sampled_frames('video.mp4', fps))
                self.assertEqual([int(frame[0, 0, 0]) for _, frame in sampled], expected)
                np.testing.assert_allclose([time for time, _ in sampled], np.array(expected) / rate)
                observation_indices, _ = sample_indices(count, rate, fps, 'video')
                self.assertEqual(observation_indices, expected)
                for streaming in (True, False):
                    model = self.compare_inputs(self.sample('video.mp4', [1, 1, 5, 100]), fps, streaming)
                    if streaming:
                        self.assertEqual(model.questions,
                                         [tuple(expected[:int(cutoff * fps)])
                                          for cutoff in (1, 1, 5, 100)])
                    else:
                        self.assertEqual(model.questions, [tuple(expected)] * 4)

    def test_original_sampling_edge_cases(self):
        np.save(self.path, self.video[:5])
        self.assertEqual(list(sampled_frames(self.path, .1)), [])
        with patch('decord.VideoReader', return_value=VideoReaderFixture(25., 5)):
            with self.assertRaisesRegex(ValueError, 'zero frame stride'):
                list(sampled_frames('video.mp4', 50.))

    def test_invalid_streaming_times_rejected_by_both(self):
        for cutoffs in ([2, 1], [-1], [float('nan')], [float('inf')]):
            sample = self.sample(self.path, cutoffs)
            analyzer = ReKVStreamVQA(anno=[sample], save_dir=self.temporary.name,
                                    sample_fps=.5, qa_model=RecordingModel())
            with self.subTest(cutoffs=cutoffs):
                with self.assertRaisesRegex(ValueError, 'chronological'):
                    analyzer.analyze_a_video(sample)
                with self.assertRaisesRegex(ValueError, 'chronological'):
                    SuPrKVStreamVQA(anno=[sample], save_dir=self.temporary.name,
                                    sample_fps=.5, qa_model=RecordingModel()).analyze_a_video(sample)


if __name__ == '__main__':
    unittest.main()
