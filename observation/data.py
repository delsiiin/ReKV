"""RVS records and decoding of sampled frames with an optional stopping count."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


class RVSDataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.records = {}
        self.annotations = {}
        for split, filename in [('ego', 'ego4d_oe.json'), ('movie', 'movienet_oe.json')]:
            path = self.root / split / filename
            if not path.is_file():
                continue
            records = json.loads(path.read_text())
            if not isinstance(records, list):
                raise ValueError(f"Expected a list in {path}")
            for index, record in enumerate(records):
                sample_id = f'{split}:{index}'
                self.records[sample_id] = dict(record, sample_id=sample_id, split=split)
                self.annotations[sample_id] = path
        if not self.records:
            raise ValueError(f"No RVS annotations found in {self.root}")

    def get(self, sample_id):
        if sample_id not in self.records:
            raise ValueError("Unknown RVS sample")
        return self.records[sample_id]

    def video_path(self, record):
        # Resolve only files underneath the chosen RVS root, ignoring old machine paths.
        basename = Path(record['video_path']).name
        path = (self.root / record['split'] / 'videos' / basename).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Video is outside the RVS root")
        if not path.is_file():
            raise ValueError(f"Video file not found: {path}")
        return path

    def summaries(self):
        return [dict(sample_id=r['sample_id'], split=r['split'], video_id=r['video_id'],
                     duration=r.get('duration'), questions=len(r.get('conversations', [])))
                for r in self.records.values()]


def sample_indices(total, source_fps, sample_fps, kind, start_time=0, max_frames=None):
    """Match ReKVStreamVQA sampling, then optionally stop after max_frames samples."""
    if total < 1 or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Invalid source video length or FPS")
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("Sample FPS must be positive")
    if not math.isfinite(start_time) or start_time < 0:
        raise ValueError("Start must be nonnegative")
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError('max_frames must be a positive integer or None')
    if kind == 'npy':
        if sample_fps > 1:
            raise ValueError("RVS movie arrays contain 1 FPS; sample_fps must be <= 1")
        count = int(total * sample_fps)
        if count == 0:
            raise ValueError("Sample FPS is too small to select any frames")
        indices = np.linspace(0, total - 1, count, dtype=int)
    else:
        step = int(round(source_fps) / sample_fps)
        if step < 1:
            raise ValueError("Sample FPS exceeds the video FPS")
        indices = np.arange(0, total, step, dtype=np.int64)
    eligible = indices[indices / source_fps >= start_time]
    if not len(eligible):
        raise ValueError("No sampled frames at or after this start time")
    return eligible[:max_frames].tolist(), len(eligible)


def decode_frames(path, params, destination):
    path, destination = Path(path), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.npy':
        source = np.load(path, mmap_mode='r', allow_pickle=False)
        if source.ndim != 4 or source.shape[-1] != 3 or source.dtype != np.uint8:
            raise ValueError("Expected RVS movie uint8 [frames, height, width, RGB]")
        fps, kind = 1.0, 'npy'
    else:
        from decord import VideoReader, cpu
        source = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        fps, kind = float(source.get_avg_fps()), 'video'
    total = len(source)
    indices, available = sample_indices(total, fps, params['sample_fps'], kind,
                                        params['start_time'], params.get('max_frames'))
    images = np.asarray(source[indices]) if kind == 'npy' else source.get_batch(indices).asnumpy()
    frames = []
    for index, (source_index, pixels) in enumerate(zip(indices, images)):
        filename = f'frame-{index:04d}.jpg'
        Image.fromarray(pixels).save(destination / filename, quality=95)
        frames.append(dict(index=index, source_index=source_index,
                           timestamp=source_index / fps, image=filename,
                           width=int(pixels.shape[1]), height=int(pixels.shape[0])))
    metadata = dict(source_frames=total, source_fps=fps, source_duration=total / fps,
                    available_sampled_frames=available, processed_frames=len(frames),
                    truncated=available > len(frames),
                    sampling_policy='ReKV np.linspace at 1 FPS' if kind == 'npy'
                    else 'ReKV range(step=int(round(source_fps)/sample_fps))',
                    first_timestamp=frames[0]['timestamp'], last_timestamp=frames[-1]['timestamp'])
    return images, frames, metadata
