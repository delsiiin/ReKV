"""Explicit RoPE positions for stored keys and position-delta rebasing."""
import torch


def rotate_at(value, positions, rope):
    """Apply R(positions), including negative deltas, without a dense position table."""
    rotary_dim = rope.inv_freq.numel() * 2
    if rotary_dim > value.shape[-1]:
        raise ValueError('RoPE dimension exceeds the head dimension')
    angles = positions.to(device=value.device, dtype=torch.float32)[..., None]
    angles = angles * rope.distance_scale * rope.inv_freq.to(value.device)
    angles = torch.cat((angles, angles), -1)
    part = value[..., :rotary_dim].float()
    first, second = part.chunk(2, -1)
    rotated = part * angles.cos() + torch.cat((-second, first), -1) * angles.sin()
    return torch.cat((rotated.to(value.dtype), value[..., rotary_dim:]), -1).contiguous()
