"""Uniform video sampling for files and the repository's 1-FPS NPY arrays.

smart_nframes/smart_resize and bicubic resizing follow qwen-vl-utils 0.0.14
(Apache-2.0). The upstream InfiniPot-V dependency was unpinned; this adapter
fixes the utility behavior without requiring a different Transformers install.
OneVision uses individual frames and delegates resizing to its own processor.
"""
import math
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import InterpolationMode, functional


def frame_indices(total_frames, video_fps, sample_fps=2., max_frames=768, frame_factor=2):
    if total_frames <= 0:
        return torch.empty(0, dtype=torch.long)
    if not math.isfinite(video_fps) or video_fps <= 0:
        raise ValueError('Source video FPS must be finite and positive')
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError('sample_fps must be finite and positive')
    # Qwen uses pairs of raw frames; OneVision consumes individual frames.
    if frame_factor not in (1, 2):
        raise ValueError('frame_factor must be 1 or 2')
    nframes = min(max(total_frames / video_fps * sample_fps, 4),
                  max_frames // frame_factor * frame_factor, total_frames)
    nframes = math.floor(nframes / frame_factor) * frame_factor
    if nframes < frame_factor:
        # Dataset extension: a one-frame visible prefix is padded like Qwen's
        # list-of-frames input, rather than reading a future frame.
        return torch.zeros(frame_factor, dtype=torch.long)
    return torch.linspace(0, total_frames - 1, nframes).round().long()


def resize_video(video):
    # InfiniPot-V sets max_pixels=128*28*28, equal to the utility's min_pixels.
    height, width = video.shape[-2:]
    factor, pixels = 28, 128 * 28 * 28
    if max(height, width) / min(height, width) > 200:
        raise ValueError('Video aspect ratio must not exceed 200')
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > pixels:
        beta = math.sqrt(height * width / pixels)
        h, w = math.floor(height / beta / factor) * factor, math.floor(width / beta / factor) * factor
    elif h * w < pixels:
        beta = math.sqrt(pixels / (height * width))
        h, w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return functional.resize(video, [h, w], interpolation=InterpolationMode.BICUBIC, antialias=True).float()


def load_video(path, *, sample_fps=2., max_frames=768, end_time=None,
               frame_factor=2, resize=True):
    """Uniformly sample a complete video or an exclusive visible prefix.

    Limit decoding before materializing RGB frames. RVS NPY files are 1 FPS;
    their annotation times already address the full stored array.
    """
    if end_time is not None and (not math.isfinite(end_time) or end_time < 0):
        raise ValueError('end_time must be finite and nonnegative')
    if Path(path).suffix.lower() == '.npy':
        source = np.load(path, mmap_mode='r')
        video_fps = 1.
        if source.ndim != 4 or source.shape[-1] != 3:
            raise ValueError('NPY videos must have shape (T, H, W, 3)')
        read = lambda indices: np.asarray(source[indices])
    else:
        from decord import VideoReader, cpu
        source = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        video_fps = source.get_avg_fps()
        read = lambda indices: source.get_batch(indices).asnumpy()
    total = len(source)
    if end_time is not None:
        # Exclusively include source timestamps < end_time, never a future frame.
        total = min(total, math.ceil(end_time * video_fps))
    indices = frame_indices(total, video_fps, sample_fps, max_frames, frame_factor)
    if not len(indices):
        if end_time is None:
            raise ValueError(f'Video has no frames: {path}')
        return None
    video = torch.from_numpy(read(indices.tolist()).copy()).permute(0, 3, 1, 2)
    return resize_video(video) if resize else video
