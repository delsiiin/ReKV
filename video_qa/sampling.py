"""Dataset selection and video frame sampling for evaluation workers."""
import argparse
import json
import math


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def load_samples(anno_path, max_samples=0):
    """Select the first max_samples video records; zero means all records."""
    if max_samples < 0:
        raise ValueError("max_samples must be nonnegative")
    with open(anno_path) as annotation_file:
        samples = json.load(annotation_file)
    return samples[:max_samples] if max_samples else samples


def split_samples(samples, num_chunks):
    """Split records into exactly num_chunks nonempty, contiguous chunks."""
    if not 1 <= num_chunks <= len(samples):
        raise ValueError("num_chunks must be between 1 and the number of selected samples")
    size, remainder = divmod(len(samples), num_chunks)
    chunks = []
    start = 0
    for index in range(num_chunks):
        end = start + size + (index < remainder)
        chunks.append(samples[start:end])
        start = end
    return chunks


def sampled_frames(path, fps=1.):
    """Yield timestamped frames selected by the original ReKV sampling rules.

    NPY inputs are 1-FPS arrays sampled with linspace. Other videos use
    int(round(source_fps) / fps) as a fixed frame-index stride. Timestamps
    still refer to source indices, using the unrounded average video FPS.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive')
    if str(path).endswith('.npy'):
        import numpy as np

        if fps > 1:
            raise ValueError('Pre-sampled NPY videos support at most 1 FPS')
        video = np.load(path, mmap_mode='r')
        indices = np.linspace(0, len(video) - 1, int(len(video) * fps), dtype=int)
        for index in indices:
            yield float(index), np.asarray(video[index]).copy()
    else:
        from decord import VideoReader, cpu

        video = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        native_fps = video.get_avg_fps()
        if not math.isfinite(native_fps) or native_fps <= 0:
            raise ValueError('Source video FPS must be finite and positive')
        stride = int(round(native_fps) / fps)
        if stride < 1:
            raise ValueError('Sample FPS produces a zero frame stride')
        for index in range(0, len(video), stride):
            yield index/native_fps, video[index].asnumpy()
