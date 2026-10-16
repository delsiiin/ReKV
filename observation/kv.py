"""Immutable prefill K/V snapshots and full-resolution magnitude figures."""
from pathlib import Path

import numpy as np


def save_kv_snapshot(cache, frames, directory, check_cancel=lambda: None, on_layer=lambda n, total: None):
    """Persist native, unexpanded KV heads as [K/V, frame, head, token, channel]."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not frames or not getattr(cache, 'layers', None):
        raise ValueError('K/V snapshot requires frames and a native Transformers cache')
    start, end = frames[0]['token_start'], frames[-1]['token_end']
    tokens = frames[0]['token_end'] - start
    if tokens < 1 or any(f['token_start'] != start + i * tokens or f['token_end'] != start + (i + 1) * tokens
                         for i, f in enumerate(frames)):
        raise ValueError('Video frame token ranges must be contiguous and equal-sized')
    shape = None
    source_dtype = storage_dtype = None
    for index, layer in enumerate(cache.layers):
        check_cancel()
        key, value = layer.keys, layer.values
        if key.shape != value.shape or key.ndim != 4 or key.shape[0] != 1 or end > key.shape[2]:
            raise ValueError('Invalid native K/V cache shape or video token range')
        expected = (2, len(frames), key.shape[1], tokens, key.shape[-1])
        if shape is not None and shape != expected:
            raise ValueError('K/V shapes differ between decoder layers')
        shape, source_dtype = expected, str(key.dtype).removeprefix('torch.')
        # NumPy cannot represent bfloat16; float32 represents every bfloat16 value exactly.
        storage_dtype = np.dtype('float16' if source_dtype == 'float16' else 'float32')
        temporary = directory / f'layer-{index:03d}.tmp'
        mapped = np.lib.format.open_memmap(temporary, mode='w+', dtype=storage_dtype, shape=shape)
        try:
            for kind, tensor in enumerate((key, value)):
                check_cancel()
                selected = tensor.detach()[0, :, start:end].cpu()
                if source_dtype != 'float16':
                    selected = selected.float()
                array = selected.numpy().reshape(shape[2], len(frames), tokens, shape[-1]).transpose(1, 0, 2, 3)
                if not np.isfinite(array).all():
                    raise ValueError('Non-finite K/V cache values')
                mapped[kind] = array
            mapped.flush()
        finally:
            del mapped
        temporary.replace(directory / f'layer-{index:03d}.npy')
        on_layer(index + 1, len(cache.layers))
    return dict(layers=len(cache.layers), frames=len(frames), kv_heads=shape[2],
                tokens_per_frame=tokens, channels=shape[-1], source_dtype=source_dtype,
                storage_dtype=str(storage_dtype), storage_layout=['key_value', 'frame', 'kv_head', 'token', 'channel'],
                stored_values='signed native K/V; displayed magnitude = elementwise absolute value',
                snapshot_phase='observation prefill, before answer generation', key_position='post-RoPE',
                includes_window_tokens=True, includes_text_tokens=False, layer_indices='zero-based',
                head_indices='native KV heads, zero-based; no GQA repetition or head averaging',
                data_bytes=int(np.prod(shape)) * storage_dtype.itemsize * len(cache.layers))


def read_kv_slice(directory, manifest, layer, frame, head):
    for value, size, label in [(layer, manifest['layers'], 'layer'), (frame, manifest['frames'], 'frame'),
                               (head, manifest['kv_heads'], 'KV head')]:
        if type(value) is not int or not 0 <= value < size:
            raise ValueError(f'Invalid {label} index')
    path = Path(directory) / f'layer-{layer:03d}.npy'
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    expected = (2, manifest['frames'], manifest['kv_heads'], manifest['tokens_per_frame'], manifest['channels'])
    if array.shape != expected:
        raise ValueError('Stored K/V shape does not match the snapshot metadata')
    # Copy only the selected frame/head. The complete video is never loaded here.
    return np.array(array[:, frame, head], dtype=np.float32, copy=True)


def magnitude_payload(pair, layer, frame, head, token_start):
    if pair.ndim != 3 or pair.shape[0] != 2 or not np.isfinite(pair).all():
        raise ValueError('Expected finite K/V tensors of shape [2, tokens, channels]')
    magnitude = np.abs(pair)
    payload = dict(layer=layer, frame=frame, head=head, token_start=token_start,
                   tokens=pair.shape[1], channels=pair.shape[2], layout='[token][channel]')
    for name, array in zip(('key', 'value'), magnitude):
        payload[name] = dict(magnitude=array.tolist(), maximum=float(array.max()),
                             mean=float(array.mean()), p99=float(np.quantile(array, .99)),
                             token_mean=array.mean(axis=1).tolist(), channel_mean=array.mean(axis=0).tolist())
    return payload


def render_kv_surface(pair, destination, *, model_name, layer, head, frame, elevation=30, azimuth=-60, shared_scale=False):
    """Save a standalone scientific plot using every token/channel (no downsampling)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    magnitude = np.abs(np.asarray(pair, dtype=np.float32))
    tokens, channels = magnitude.shape[1:]
    token_grid, channel_grid = np.meshgrid(np.arange(tokens), np.arange(channels))
    figure = Figure(figsize=(12, 5.2), dpi=135, layout='constrained')
    FigureCanvasAgg(figure)
    for index, name in enumerate(('Key', 'Value')):
        axis = figure.add_subplot(1, 2, index + 1, projection='3d')
        values = magnitude[index]
        maximum = float(magnitude.max() if shared_scale else values.max())
        maximum = max(maximum, 1e-9)
        axis.plot_surface(token_grid, channel_grid, values.T, rcount=channels, ccount=tokens,
                          cmap='coolwarm', norm=Normalize(vmin=0, vmax=maximum), linewidth=0, antialiased=False)
        axis.set_title(f'{model_name}\nLayer {layer} / KV head {head} / Frame {frame + 1} / {name} Cache', fontsize=10, pad=12)
        axis.set_xlabel('Token (within frame)', fontsize=10, labelpad=7)
        axis.set_ylabel('Channel', fontsize=10, labelpad=7)
        axis.set_zlabel('Absolute Value', fontsize=10, labelpad=5)
        axis.set_xlim(0, max(tokens - 1, 1))
        axis.set_ylim(0, max(channels - 1, 1))
        axis.set_zlim(0, maximum * 1.04)
        axis.set_xticks(np.unique(np.linspace(0, tokens - 1, min(tokens, 5), dtype=int)))
        axis.set_yticks(np.unique(np.linspace(0, channels - 1, min(channels, 5), dtype=int)))
        axis.zaxis.set_major_locator(MaxNLocator(5))
        axis.tick_params(labelsize=8, pad=1)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_box_aspect((1.25, 1, .8))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    figure.savefig(temporary, format='png', dpi=135, facecolor='white')
    figure.clear()
    temporary.replace(destination)
