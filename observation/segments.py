"""Vision-feature segmentation and Key/hidden-state cosine observations."""
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch


EPSILON = 1e-6
SUMMARY_SYSTEM = (
    'Identify the main participants in a video segment and describe their visible '
    'actions concisely. Base your summary only on the provided frames.'
)
SUMMARY_PROMPT = (
    'Summarize the main entity\'s behavior in this segment in exactly one short sentence. '
    'Focus on the entity or tightly interacting entities most involved in the main action, '
    'prioritizing sustained or repeated participation. State who or what is doing what. '
    'Output only that factual sentence, without analysis, explanation, timestamps, '
    'or background details. If no action is visible, briefly describe the main visible '
    'entity\'s state without inventing an action.'
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
def capture_queries(decoder, callback, check_cancel=lambda: None):
    """Capture the model's actual q_proj outputs before RoPE."""
    handles = []
    try:
        for index, layer in enumerate(decoder.layers):
            def hook(module, inputs, output, index=index):
                check_cancel()
                callback(index, output)
            handles.append(layer.self_attn.q_proj.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def capture_token_states(decoder, callback, check_cancel=lambda: None, *, include_queries=False):
    with capture_keys(decoder, lambda layer, x: callback('key', layer, x), check_cancel), \
            capture_hidden_states(decoder, lambda layer, x: callback('hidden', layer, x), check_cancel), \
            (capture_queries(decoder, lambda layer, x: callback('query', layer, x), check_cancel)
             if include_queries else nullcontext()):
        yield


@contextmanager
def capture_rotary_states(decoder, callback, check_cancel=lambda: None):
    """Observe native post-RoPE Q/K at SDPA dispatch without changing the forward."""
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import AttentionMaskInterface

    name = 'rvs_rotary_state_observer'
    def attention(module, query, key, value, attention_mask, **kwargs):
        check_cancel()
        callback(module.layer_idx, query, key)
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

    AttentionInterface.register(name, attention)
    AttentionMaskInterface.register(name, AttentionMaskInterface()['sdpa'])
    previous = decoder.config._attn_implementation
    decoder.config._attn_implementation = name
    try:
        yield
    finally:
        decoder.config._attn_implementation = previous


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


def segment_by_ema(distances, frames, alpha=.2, min_frames=1):
    """Use the prior EMA as threshold; first pair bootstraps without a boundary."""
    values = np.asarray(distances, dtype=np.float64)
    if values.shape != (len(frames) - 1,) or not len(frames):
        raise ValueError('Expected one distance per adjacent frame pair')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Distances must be finite and nonnegative')
    if not 0 < alpha <= 1 or type(min_frames) is not int or min_frames < 1:
        raise ValueError('Invalid EMA parameters')
    start, ema, segments, transitions = 0, None, [], []
    for index, distance in enumerate(values, 1):
        threshold = None if ema is None else max(EPSILON, ema)
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
                alpha=alpha, multiplier=1.0, min_frames=min_frames)


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


STATE_DIRECTORIES = {'key': 'token_keys', 'hidden': 'token_hidden', 'query': 'token_queries',
                     'key_rope': 'token_keys_rope', 'query_rope': 'token_queries_rope'}


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
    if representation not in ('key', 'hidden'):
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


def mean_segment_attention(query, key, heads, kv_heads, query_chunk=32, key_mask=None):
    """QK diagnostic; apply the explicit key mask before softmax per head/query.

    Return separate means of logits and probabilities. Do not softmax mean logits.
    A single head and a bounded query chunk are materialized at a time.
    """
    query, key = np.asarray(query, dtype=np.float64), np.asarray(key, dtype=np.float64)
    if (query.ndim != 2 or key.ndim != 2 or min(query.shape) < 1 or min(key.shape) < 1
            or heads < 1 or kv_heads < 1 or heads % kv_heads or query_chunk < 1
            or query.shape[1] % heads or key.shape[1] % kv_heads
            or query.shape[1] // heads != key.shape[1] // kv_heads):
        raise ValueError('Invalid Q/K dimensions or GQA head configuration')
    if not np.isfinite(query).all() or not np.isfinite(key).all():
        raise ValueError('Attention requires finite Q/K states')
    if key_mask is None:
        key_mask = np.ones(len(key), dtype=bool)
    key_mask = np.asarray(key_mask)
    if key_mask.shape != (len(key),) or key_mask.dtype != np.bool_ or not key_mask.any():
        raise ValueError('Key mask must be boolean and retain at least one key')
    dim = query.shape[1] // heads
    q, k = query.reshape(-1, heads, dim), key.reshape(-1, kv_heads, dim)
    before, after = np.zeros(len(key)), np.zeros(len(key))
    for head in range(heads):
        kh = k[:, head // (heads // kv_heads), :]
        for start in range(0, len(query), query_chunk):
            logits = q[start:start + query_chunk, head, :] @ kh.T / np.sqrt(dim)
            logits[:, ~key_mask] = -np.inf
            before += logits.sum(0)
            probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            after += probabilities.sum(0)
    return before / (heads * len(query)), after / (heads * len(query))


def segment_attention(directory, result, summary_index, segment_index, frame_index, layer, rope='pre'):
    experiment = result.get('segment_experiment', {})
    if 'segment_attention' not in experiment:
        raise ValueError('Summary Queries are unavailable; regenerate the observation')
    segments = experiment['segments']
    for value, upper, name in [(summary_index, len(segments), 'summary'),
                                (segment_index, len(segments), 'segment'),
                                (frame_index, len(result['frames']), 'frame')]:
        if type(value) is not int or not 0 <= value < upper:
            raise ValueError(f'Invalid {name} index')
    segment = segments[segment_index]
    if frame_index not in segment['frame_indices']:
        raise ValueError('Selected frame is outside the selected segment')
    layers = result['model']['layers']
    if layer == 'all':
        selected = range(layers)
    elif isinstance(layer, str) and layer.isascii() and layer.isdigit() and 0 <= int(layer) < layers:
        selected = [int(layer)]
    else:
        raise ValueError('Invalid layer')
    if rope not in ('pre', 'post', 'both'):
        raise ValueError('Invalid RoPE mode; expected pre, post or both')
    modes = ['pre', 'post'] if rope == 'both' else [rope]
    available = experiment['segment_attention'].get('rope_modes', ['pre'])
    if any(mode not in available for mode in modes):
        raise ValueError('Post-RoPE states are unavailable; regenerate the observation')
    tokens = segments[summary_index]['summary']['tokens']
    count = result['sequence']['tokens_per_frame']
    first, last = segment['start_frame'], segment['end_frame']
    segment_tokens = (last - first) * count
    offset = (frame_index - first) * count
    # The logical key sequence is [all selected segment tokens, all summary tokens].
    # Every summary column (including the Query token itself) is masked explicitly.
    key_mask = np.arange(segment_tokens + len(tokens)) < segment_tokens
    variants = []
    for mode in modes:
        before, after = np.zeros(segment_tokens), np.zeros(segment_tokens)
        masked_mass = 0.0
        qdir = 'token_queries' if mode == 'pre' else 'token_queries_rope'
        kdir = 'token_keys' if mode == 'pre' else 'token_keys_rope'
        if tokens:
            for index in selected:
                q = np.load(Path(directory) / qdir / f'summary-{summary_index}-{index}.npy', mmap_mode='r', allow_pickle=False)
                k = np.load(Path(directory) / kdir / f'video-{index}.npy', mmap_mode='r', allow_pickle=False)[first:last]
                summary_k = np.load(Path(directory) / kdir / f'summary-{summary_index}-{index}.npy', mmap_mode='r', allow_pickle=False)
                if (q.shape[0] != len(tokens) or k.shape[:2] != (last - first, count)
                        or summary_k.shape != (len(tokens), k.shape[-1])):
                    raise ValueError('Stored Q/K dimensions do not match the result')
                combined_k = np.concatenate([k.reshape(-1, k.shape[-1]), summary_k], axis=0)
                logits, probabilities = mean_segment_attention(q, combined_k,
                    result['model']['heads'], result['model']['kv_heads'], key_mask=key_mask)
                before += logits[:segment_tokens]
                after += probabilities[:segment_tokens]
                masked_mass += float(probabilities[segment_tokens:].sum())
            before /= len(selected)
            after /= len(selected)
            masked_mass /= len(selected)
        variants.append(dict(rope=mode,
            pre_softmax=before[offset:offset + count].tolist() if tokens else [None] * count,
            post_softmax=after[offset:offset + count].tolist() if tokens else [None] * count,
            frame_masses=[dict(frame_index=first + i, mass=float(row.sum()) if tokens else None)
                          for i, row in enumerate(after.reshape(-1, count))],
            segment_mass=float(after.sum()) if tokens else None,
            summary_mass=masked_mass if tokens else None))
    return dict(summary_index=summary_index, segment_index=segment_index, frame_index=frame_index, layer=layer,
                text_tokens=tokens, image_tokens=count, segment_tokens=segment_tokens,
                frame_token_start=result['frames'][frame_index]['token_start'],
                **variants[0], variants=variants, requested_rope=rope,
                masked_summary_tokens=len(tokens), total_key_tokens=segment_tokens + len(tokens),
                mask='segment columns visible; ALL summary key columns set to -inf before softmax; prompt and other segments excluded',
                normalization='softmax over visible segment keys separately per summary token and query head',
                aggregation='mean of logits and mean of probabilities separately over summary tokens, query heads and selected layers',
                positional_encoding='pre/post native RoPE comparison using the original position IDs of each independent input',
                positions=dict(video_token_start=result['frames'][first]['token_start'],
                               video_token_end=result['frames'][last-1]['token_end'],
                               summary_token_start=segments[summary_index]['summary'].get('encoding_token_start'),
                               summary_token_end=segments[summary_index]['summary'].get('encoding_token_end')))
