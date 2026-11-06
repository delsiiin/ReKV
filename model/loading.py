"""Shared single-device weight placement for video QA backends."""
import torch


def load_pretrained(model_class, model_path, *, device=None, **kwargs):
    """Load all weights on one device; eval workers see only their assigned GPU."""
    device = torch.device(device or 'cuda:0')
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda:0')
    return model_class.from_pretrained(
        model_path, device_map=str(device), low_cpu_mem_usage=True, **kwargs)
