"""Qwen2.5-VL frame encoding for ReKV, also reused by SuPr-KV.

Each frame is a one-frame video, padded by the HF processor to a temporal
patch. Frames must never share a temporal patch: retrieval IDs refer to sampled
frames, and streaming must work without looking ahead to the next frame.
"""
import torch
from torchvision.transforms.functional import resize, InterpolationMode


FRAME_GRID = 14
FRAME_TOKENS = FRAME_GRID ** 2


def frame_features(model, processor, frames):
    """Return (frames, 196, hidden) using the checkpoint's video vision encoder.

    Inputs follow the repository video convention: uint8 RGB (N, H, W, C).
    Explicit square resizing gives fixed cache blocks for any aspect ratio.
    """
    frames = torch.as_tensor(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] == 0:
        raise ValueError('Expected nonempty RGB frames with shape (N, H, W, 3)')
    vision = model.config.vision_config
    side = FRAME_GRID * vision.patch_size * vision.spatial_merge_size
    frames = resize(frames.permute(0, 3, 1, 2), [side, side],
                    interpolation=InterpolationMode.BICUBIC, antialias=True)
    batch = processor.video_processor(
        videos=[frame.unsqueeze(0) for frame in frames],
        input_data_format='channels_first', do_resize=False,
        do_sample_frames=False, return_tensors='pt')
    grid = batch.video_grid_thw
    expected = torch.tensor([1, FRAME_GRID * vision.spatial_merge_size,
                             FRAME_GRID * vision.spatial_merge_size])
    if grid.shape != (len(frames), 3) or not torch.all(grid.cpu() == expected):
        raise ValueError(f'Unexpected Qwen2.5-VL frame grid: {grid.tolist()}')
    visual = model.model.visual
    output = model.model.get_video_features(
        pixel_values_videos=batch.pixel_values_videos.to(visual.device, visual.dtype),
        video_grid_thw=grid.to(visual.device), return_dict=True)
    features = torch.stack(output.pooler_output)
    if features.shape[:2] != (len(frames), FRAME_TOKENS):
        raise ValueError(f'Unexpected Qwen2.5-VL features: {tuple(features.shape)}')
    return features.to(model.get_input_embeddings().weight.device)
