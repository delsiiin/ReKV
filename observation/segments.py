"""Vision-feature segmentation and Key/hidden-state cosine observations."""
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


EPSILON = 1e-6
SUMMARY_SYSTEM = (
    'You are a careful streaming-video observer. Actively track visible objects, people, '
    'actions, on-screen text and numbers, temporal changes, and anomalies. Retain '
    'distinctive visual evidence for later sequential questions. Describe only the '
    'provided segment; do not import events from other segments or invent unseen details.'
)
SUMMARY_PROMPT = (
    'Summarize this event segment in a concise factual paragraph (about 60–100 words). '
    'Prioritize: (1) key objects/people, attributes, counts and spatial layout; '
    '(2) actions, interactions and visible state transitions in chronological order; '
    '(3) readable on-screen text, numbers, labels or timers; '
    '(4) outcomes, unusual events and clues that distinguish this segment from similar '
    'events. Mention causal links only when directly supported by the images. '
    'Use the supplied sampled-frame timestamps when locating a visible change; do not '
    'infer exact event times between frames. If a detail is unclear, say so briefly. '
    'Omit absent categories. Output only the summary, without answering a question.'
)


def summary_prompt(frames):
    times = ', '.join(f"{frame['timestamp']:.3f}s" for frame in frames)
    return f'{SUMMARY_PROMPT}\nSampled-frame timestamps, in image order: {times}.'


@contextmanager
def capture_keys(decoder, callback, check_cancel=lambda: None):
    """Capture k_proj output before head reshape/RoPE, without modifying outputs."""
    handles = []
    try:
        for index, layer in enumerate(decoder.layers):
            def hook(module, inputs, output, index=index):
                check_cancel()
                callback(index, output)
            handles.append(layer.self_attn.k_proj.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def capture_hidden_states(decoder, callback, check_cancel=lambda: None):
    """Decoder block outputs, after both residuals and before final model norm."""
    handles = []
    try:
        for index, layer in enumerate(decoder.layers):
            def hook(module, inputs, output, index=index):
                check_cancel()
                callback(index, output[0] if isinstance(output, tuple) else output)
            handles.append(layer.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def capture_token_states(decoder, callback, check_cancel=lambda: None):
    with capture_keys(decoder, lambda layer, x: callback('key', layer, x), check_cancel), \
            capture_hidden_states(decoder, lambda layer, x: callback('hidden', layer, x), check_cancel):
        yield


def selected_vision_features(output, layer, strategy):
    """Same feature-layer selection as OneVision, before projector and pooling."""
    if strategy not in ('full', 'default'):
        raise ValueError('Unsupported vision feature selection strategy')
    features = output.hidden_states[layer] if isinstance(layer, int) else torch.cat(
        [output.hidden_states[index] for index in layer], dim=-1)
    return features[:, 1:] if strategy == 'default' else features


def vision_frame_distances(features, previous=None, epsilon=EPSILON):
    """Aligned raw vision patches; carry the last frame across encoder batches."""
    if features.ndim != 3 or any(size < 1 for size in features.shape):
        raise ValueError('Expected nonempty vision features [frames, patches, dim]')
    distances = []
    for current in features:
        current = current.float()
        if previous is not None:
            if current.shape != previous.shape:
                raise ValueError('Vision patch grids or dimensions changed between frames')
            scale = (current.square().sum(-1).mean() + epsilon).sqrt()
            distances.append(float((current - previous).norm(dim=-1).mean() / scale))
        previous = current
    # A slice would keep the whole encoder batch alive.
    return np.asarray(distances), previous.detach().clone()


def segment_by_ema(distances, frames, alpha=.2, multiplier=1.5, min_frames=1):
    """Use the prior EMA as threshold; first pair bootstraps without a boundary."""
    values = np.asarray(distances, dtype=np.float64)
    if values.shape != (len(frames) - 1,) or not len(frames):
        raise ValueError('Expected one distance per adjacent frame pair')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Distances must be finite and nonnegative')
    if not 0 < alpha <= 1 or multiplier <= 0 or type(min_frames) is not int or min_frames < 1:
        raise ValueError('Invalid EMA parameters')
    start, ema, segments, transitions = 0, None, [], []
    for index, distance in enumerate(values, 1):
        threshold = None if ema is None else max(EPSILON, multiplier * ema)
        exceeds = threshold is not None and distance > threshold
        boundary = bool(exceeds and index - start >= min_frames)
        transitions.append(dict(previous_frame=index - 1, current_frame=index,
                                distance=float(distance), similarity=float(1 / (1 + distance)),
                                ema_before=ema, threshold=threshold, boundary=boundary,
                                reason='warmup' if threshold is None else
                                'boundary' if boundary else 'min_frames' if exceeds else 'below_threshold'))
        if boundary:
            segments.append(dict(start_frame=start, end_frame=index))
            start = index
        ema = float(distance) if ema is None else float((1 - alpha) * ema + alpha * distance)
        transitions[-1]['ema_after'] = ema
    segments.append(dict(start_frame=start, end_frame=len(frames)))
    for index, segment in enumerate(segments):
        selected = frames[segment['start_frame']:segment['end_frame']]
        segment.update(id=index, frame_indices=[f['index'] for f in selected],
                       first_timestamp=selected[0]['timestamp'], last_timestamp=selected[-1]['timestamp'])
    return dict(segments=segments, transitions=transitions,
                alpha=alpha, multiplier=multiplier, min_frames=min_frames)


def cosine_matrix(text, visual):
    """Undefined (empty or zero-norm) embeddings are null, never fabricated zeros."""
    text, visual = np.asarray(text, dtype=np.float64), np.asarray(visual, dtype=np.float64)
    numerator = text @ visual.T
    denominator = np.linalg.norm(text, axis=-1)[:, None] * np.linalg.norm(visual, axis=-1)[None, :]
    valid = (denominator > 0) & np.isfinite(denominator) & np.isfinite(numerator)
    output = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=output, where=valid)
    output = np.clip(output, -1, 1)
    return [[float(x) if np.isfinite(x) else None for x in row] for row in output]


STATE_DIRECTORIES = {'key': 'token_keys', 'hidden': 'token_hidden'}


class TokenStateStore:
    """Persist unpooled token states one layer at a time."""
    def __init__(self, directory, representation='key'):
        if representation not in STATE_DIRECTORIES:
            raise ValueError('Invalid representation')
        self.root = Path(directory) / STATE_DIRECTORIES[representation]
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, name, keys):
        keys = keys.detach().cpu()
        if keys.dtype == torch.bfloat16:
            keys = keys.float()  # NumPy has no native bfloat16 storage.
        np.save(self.root / name, keys.numpy(), allow_pickle=False)

    def video(self, layer, keys, frames):
        count = frames[0]['token_end'] - frames[0]['token_start']
        selected = keys[0, frames[0]['token_start']:frames[-1]['token_end']]
        self.write(f'video-{layer}.npy', selected.reshape(len(frames), count, -1))

    def summary(self, layer, index, keys):
        self.write(f'summary-{index}-{layer}.npy', keys)


def token_similarity(directory, result, summary_index, frame_index, layer, representation='key'):
    """Read just one frame and summary; rows=text tokens, columns=visual tokens."""
    experiment = result.get('segment_experiment', {})
    if 'token_similarity' not in experiment:
        raise ValueError('Token states are unavailable; regenerate the observation')
    available = experiment['token_similarity'].get('representations', ['key'])
    if representation not in STATE_DIRECTORIES:
        raise ValueError('Invalid representation; expected key or hidden')
    if representation not in available:
        raise ValueError('Requested token states are unavailable; regenerate the observation')
    segments = experiment['segments']
    if type(summary_index) is not int or not 0 <= summary_index < len(segments):
        raise ValueError('Invalid summary index')
    if type(frame_index) is not int or not 0 <= frame_index < len(result['frames']):
        raise ValueError('Invalid frame index')
    layers = result['model']['layers']
    if layer == 'all':
        selected = range(layers)
    elif isinstance(layer, str) and layer.isascii() and layer.isdigit() and 0 <= int(layer) < layers:
        selected = [int(layer)]
    else:
        raise ValueError('Invalid layer')
    tokens = segments[summary_index]['summary']['tokens']
    columns = result['sequence']['tokens_per_frame']
    total = np.zeros((len(tokens), columns), dtype=np.float64)
    root = Path(directory) / STATE_DIRECTORIES[representation]
    if tokens:
        for index in selected:
            visual = np.load(root / f'video-{index}.npy', mmap_mode='r', allow_pickle=False)[frame_index]
            text = np.load(root / f'summary-{summary_index}-{index}.npy', mmap_mode='r', allow_pickle=False)
            if visual.shape != (columns, text.shape[-1]) or text.shape[0] != len(tokens):
                raise ValueError('Stored token state dimensions do not match the result')
            # Nulls propagate to the all-layer mean; no token is dropped or pooled.
            total += np.asarray(cosine_matrix(text, visual), dtype=np.float64)
        total /= len(selected)
    matrix = [[float(v) if np.isfinite(v) else None for v in row] for row in total]
    return dict(summary_index=summary_index, frame_index=frame_index, layer=layer, representation=representation,
                text_tokens=tokens, image_tokens=columns, cosine=matrix,
                frame_token_start=result['frames'][frame_index]['token_start'])
