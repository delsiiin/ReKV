"""Within-frame normalized K/V nearest-neighbor distances and redundancy ratio."""
import argparse
import csv
import io
import json
from pathlib import Path
import time

import numpy as np


METRIC = 'within_frame_separate_kv_nearest_neighbor_redundancy_mean_threshold'
DEFAULT_EPSILON = 1e-8
KEY_POSITIONS = {'pre': 'pre-RoPE', 'post': 'post-RoPE'}


def validate_epsilon(epsilon):
    if (isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, (int, float, np.integer, np.floating))
            or not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError('epsilon must be a finite positive number')


def validate_state(state):
    if state not in ('key', 'value'):
        raise ValueError('State must be key or value')


def key_uniqueness(vectors, *, epsilon=DEFAULT_EPSILON):
    """Nearest-neighbor normalized distance for one state (Key or Value) independently."""
    validate_epsilon(epsilon)
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or min(vectors.shape) < 1 or not np.isfinite(vectors).all():
        raise ValueError('Expected finite, nonempty [token, channel] states')
    n = len(vectors)
    scale = float(np.sqrt(np.mean(np.sum(vectors * vectors, axis=1))) + epsilon)
    result = dict(scale=scale, epsilon=float(epsilon),
                  zero_norm_tokens=int(np.count_nonzero(~np.any(vectors, axis=1))))
    if n < 2:
        return dict(result, uniqueness=np.full(n, np.nan), nearest_token=np.full(n, -1, dtype=np.int64),
                    undefined_reason='no_other_tokens')
    nearest = np.empty(n, dtype=np.int64)
    distances = np.empty(n)
    for start in range(0, n, 32):
        stop = min(start + 32, n)
        distance = np.linalg.norm(vectors[start:stop, None] - vectors[None, :], axis=-1) / scale
        rows = np.arange(stop - start)
        distance[rows, np.arange(start, stop)] = np.inf
        indices = distance.argmin(axis=1)
        nearest[start:stop] = indices
        distances[start:stop] = distance[rows, indices]
    return dict(result, uniqueness=distances, nearest_token=nearest, undefined_reason=None)


def uniqueness_stats(vectors, *, epsilon=DEFAULT_EPSILON):
    """tau = mean(d_i^NN); R_NN = mean(d_i^NN < tau), within each slice."""
    validate_epsilon(epsilon)
    result = key_uniqueness(vectors, epsilon=epsilon)
    distances = result['uniqueness']
    if result['undefined_reason']:
        return dict(result, tau=None, redundant=np.full(len(distances), np.nan), score=None,
                    uniqueness_sum=None, mean_uniqueness=None, max_uniqueness=None)
    # Centering preserves exact equality when every distance is the same.
    tau = float(distances.min() + np.mean(distances - distances.min()))
    redundant = distances < tau
    return dict(result, tau=float(tau), redundant=redundant, score=float(redundant.mean()),
                uniqueness_sum=float(distances.sum()), mean_uniqueness=tau,
                max_uniqueness=float(distances.max()))


def score_snapshot(directory, manifest, check_cancel=lambda: None, on_layer=lambda n, total: None,
                   *, value_directory=None, epsilon=DEFAULT_EPSILON):
    """Score each native layer/frame/head; pre-RoPE K uses the matching native cached V."""
    validate_epsilon(epsilon)
    started = time.monotonic()
    layers, frames, heads = (manifest[name] for name in ('layers', 'frames', 'kv_heads'))
    tokens, channels = manifest['tokens_per_frame'], manifest['channels']
    if manifest.get('key_position') not in KEY_POSITIONS.values():
        raise ValueError('Unknown Key snapshot position relative to RoPE')
    pre_rope = manifest['key_position'] == 'pre-RoPE'
    shape = (frames, heads, tokens, channels)
    expected = shape if pre_rope else (2, *shape)
    if pre_rope and value_directory is None:
        value_directory = Path(directory).parent / 'kv'
    fields = {'thresholds': 'tau', 'scores': 'score', 'uniqueness_sums': 'uniqueness_sum', 'mean_uniqueness': 'mean_uniqueness',
              'max_uniqueness': 'max_uniqueness', 'zero_norm_tokens': 'zero_norm_tokens',
              'scales': 'scale', 'undefined_reasons': 'undefined_reason'}
    values = {name: {state: [] for state in ('key', 'value')} for name in fields}
    for layer in range(layers):
        check_cancel()
        array = np.load(Path(directory) / f'layer-{layer:03d}.npy', mmap_mode='r', allow_pickle=False)
        if array.shape != expected:
            raise ValueError('Stored Key shape does not match the snapshot metadata')
        pair = np.load(Path(value_directory) / f'layer-{layer:03d}.npy', mmap_mode='r', allow_pickle=False) if pre_rope else array
        if pair.shape != (2, *shape):
            raise ValueError('Stored K/V shape does not match the snapshot metadata')
        keys = array if pre_rope else array[0]
        for state, states in [('key', keys), ('value', pair[1])]:
            layer_values = {name: [] for name in fields}
            for frame in range(frames):
                frame_values = {name: [] for name in fields}
                for head in range(heads):
                    check_cancel()
                    statistics = uniqueness_stats(states[frame, head], epsilon=epsilon)
                    for name, source in fields.items():
                        frame_values[name].append(statistics[source])
                for name in fields:
                    layer_values[name].append(frame_values[name])
            for name in fields:
                values[name][state].append(layer_values[name])
        del states
        del keys, array, pair
        on_layer(layer + 1, layers)
    return dict(
        version=5, metric=METRIC, states=['key', 'value'], score_units='ratio', epsilon=float(epsilon),
        threshold_rule='mean_nearest_neighbor_distance', threshold_formula='tau = mean_i(d_i^NN)',
        scale_formula='s_X = sqrt(mean_i(||x_i||_2^2)) + epsilon', evaluation='independent_key_and_value',
        uniqueness_formula='d_i^NN,X = min_{j != i} ||x_i-x_j||_2/s_X; X in {K,V}',
        score_formula='R_NN(tau) = mean_i(d_i^NN < tau)',
        interpretation='Fraction of tokens whose nearest-neighbor distance is strictly below the slice mean; a relative distance statistic',
        excludes_self=True, nearest_tie_policy='first index attaining the minimum distance within the selected state',
        score_layout=['state', 'layer', 'frame', 'kv_head'], layers=layers, frames=frames, kv_heads=heads,
        tokens_per_frame=tokens, channels=channels, score_bounds=[0, 1], uniqueness_bounds=[0, None],
        key_position=manifest['key_position'], value_source='native KV cache (V is not RoPE-rotated)',
        includes_window_tokens=True, includes_text_tokens=False,
        key_capture_point=manifest.get('capture_point',
                                       'pre-RoPE Key snapshot' if pre_rope else 'native KV cache after apply_rotary_pos_emb'),
        snapshot_phase=manifest['snapshot_phase'], compute_dtype='float64',
        aggregation='compute each K/V slice independently, then display the distribution across layers/heads',
        undefined_policy='score is null for fewer than two tokens; zero vectors and exact duplicates are valid',
        elapsed_seconds=time.monotonic() - started, **values,
    )


def selected_scores(metric, layer=None, head=None, state='key'):
    validate_state(state)
    for index, size, label in [(layer, metric['layers'], 'layer'), (head, metric['kv_heads'], 'KV head')]:
        if index is not None and (type(index) is not int or not 0 <= index < size):
            raise ValueError(f'Invalid uniqueness {label} index')
    values = np.asarray(metric['scores'][state], dtype=np.float64).reshape(metric['layers'], metric['frames'], metric['kv_heads'])
    if layer is not None:
        values = values[layer:layer + 1]
    if head is not None:
        values = values[:, :, head:head + 1]
    return values.transpose(1, 0, 2).reshape(metric['frames'], -1)


def score_csv(metric, frames):
    output = io.StringIO(newline='')
    writer = csv.writer(output)
    writer.writerow(['frame_index', 'source_index', 'timestamp_seconds', 'token_start', 'token_end', 'layer', 'kv_head',
                     'state', 'nearest_neighbor_redundancy', 'uniqueness_sum', 'mean_uniqueness', 'max_uniqueness',
                     'zero_norm_tokens', 'undefined_reason', 'key_position', 'metric', 'tau', 'epsilon'])
    for f, frame in enumerate(frames):
        for layer in range(metric['layers']):
            for head in range(metric['kv_heads']):
                for state in ('key', 'value'):
                    writer.writerow([f, frame['source_index'], frame['timestamp'], frame['token_start'], frame['token_end'],
                                     layer, head, state, *[metric[name][state][layer][f][head] for name in
                                     ('scores', 'uniqueness_sums', 'mean_uniqueness', 'max_uniqueness', 'zero_norm_tokens', 'undefined_reasons')],
                                     metric['key_position'], metric['metric'], metric['thresholds'][state][layer][f][head], metric['epsilon']])
    return output.getvalue()


def spatial_shape(spatial, tokens):
    if spatial is None:
        return 1, tokens
    rows, cols = spatial['rows'], spatial['cols']
    if type(rows) is not int or type(cols) is not int or min(rows, cols) < 1 or rows * cols != tokens:
        raise ValueError('Spatial grid does not match the number of Key tokens')
    return rows, cols


def uniqueness_payload(vectors, *, layer, frame, head, token_start, spatial=None, key_position='post-RoPE',
                       state='key', epsilon=DEFAULT_EPSILON):
    validate_state(state)
    if key_position not in KEY_POSITIONS.values():
        raise ValueError('Invalid Key position')
    statistics = uniqueness_stats(vectors, epsilon=epsilon)
    rows, cols = spatial_shape(spatial, len(vectors))
    payload = dict(layer=layer, frame=frame, head=head, tokens=len(vectors), token_start=token_start,
                   rows=rows, cols=cols, spatial_grid=spatial is not None, layout='row-major spatial tokens',
                   key_position=key_position, state=state, uniqueness_bounds=[0, None], metric=METRIC, score_units='ratio')
    for name, value in statistics.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.bool_:
                payload[name] = value.tolist()
            elif name == 'nearest_token':
                payload[name] = [int(i) if i >= 0 else None for i in value]
            else:
                payload[name] = [float(v) if np.isfinite(v) else None for v in value]
        else:
            payload[name] = value
    return payload


def render_uniqueness_map(vectors, destination, *, layer, frame, head, spatial=None, scale='frame',
                          key_position='post-RoPE', state='key', epsilon=DEFAULT_EPSILON):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure

    if scale not in ('frame', 'fixed'):
        raise ValueError('Invalid uniqueness color scale')
    validate_state(state)
    if key_position not in KEY_POSITIONS.values():
        raise ValueError('Invalid Key position')
    statistics = uniqueness_stats(vectors, epsilon=epsilon)
    rows, cols = spatial_shape(spatial, len(vectors))
    maximum = statistics['max_uniqueness']
    cap = maximum if scale == 'frame' and maximum is not None and maximum > 0 else 2.
    cmap = LinearSegmentedColormap.from_list('uniqueness', ['#fff7ec', '#be370f']).with_extremes(bad='#d5dce0')
    figure = Figure(figsize=(7, 6), dpi=160, layout='constrained')
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    im = axis.imshow(np.ma.masked_invalid(statistics['uniqueness'].reshape(rows, cols)),
                     cmap=cmap, vmin=0, vmax=cap, origin='upper', interpolation='nearest')
    axis.set_xlabel('Spatial token column' if spatial else 'Token index')
    axis.set_ylabel('Spatial token row' if spatial else 'Row')
    axis.set_xticks(np.unique(np.linspace(0, cols - 1, min(cols, 8), dtype=int)))
    axis.set_yticks(np.unique(np.linspace(0, rows - 1, min(rows, 8), dtype=int)))
    figure.colorbar(im, ax=axis, fraction=.046, pad=.04).set_label(f'{state.title()} nearest-neighbor normalized distance (excluding self)')
    score = 'undefined' if statistics['score'] is None else f'{statistics["score"]:.3f}'
    total = 'undefined' if statistics['uniqueness_sum'] is None else f'{statistics["uniqueness_sum"]:.6g}'
    tau = 'undefined' if statistics['tau'] is None else f'{statistics["tau"]:.6g}'
    axis.set_title(f'Frame {frame + 1} | Layer {layer} | KV head {head}\nR_NN(tau=mean={tau}) = {score} | Sum of distances = {total}', fontsize=10)
    figure.suptitle(f'Within-frame {state.title()} nearest-neighbor distance ({key_position})', fontsize=12)
    _save_figure(figure, destination)


def render_scores(metric, frames, destination, *, layer=None, head=None, frame=0, state='key'):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    data = selected_scores(metric, layer, head, state)
    count = metric['frames']
    if type(frame) is not int or not 0 <= frame < count:
        raise ValueError('Invalid uniqueness frame index')
    figure = Figure(figsize=(max(9, min(20, count * .5)), 4.6), dpi=150, layout='constrained')
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    groups = [row[np.isfinite(row)] for row in data]
    valid = [i for i, group in enumerate(groups) if group.size]
    positions = np.arange(count) + 1
    if valid:
        axis.boxplot([groups[i] for i in valid], positions=positions[valid], widths=.55, whis=(0, 100),
                     showfliers=False, patch_artist=True, manage_ticks=False,
                     boxprops=dict(facecolor='#087f80', alpha=.25), medianprops=dict(color='#087f80'))
        axis.plot(positions, [g.mean() if g.size else np.nan for g in groups], 'o-', color='#087f80', markersize=3)
    for i, group in enumerate(groups):
        if not group.size:
            axis.text(i + 1, 1, 'N/A', ha='center', fontsize=8)
    axis.set_ylim(0, 1.05)
    axis.set_xlim(.5, count + .5)
    axis.set_ylabel('Nearest-neighbor redundancy R_NN')
    axis.axvspan(frame + .6, frame + 1.4, color='#087f80', alpha=.04)
    axis.grid(axis='y', alpha=.2)
    axis.spines[['top', 'right']].set_visible(False)
    ticks = np.unique(np.linspace(0, count - 1, min(count, 12), dtype=int))
    axis.set_xticks(ticks + 1, [f'{i + 1}\n{frames[i]["timestamp"]:.2f}s' for i in ticks])
    axis.set_xlabel('Sampled frame / source timestamp')
    axis.legend(handles=[Line2D([0], [0], color='#087f80', marker='o', label='Mean'),
                         Line2D([0], [0], color='#087f80', linewidth=5, alpha=.25, label='Q1–Q3; median; whiskers: min–max')],
                fontsize=8, frameon=False)
    scope = f'Layer {layer}' if layer is not None else 'All layers'
    scope += f' / KV head {head}' if head is not None else ' / All native KV heads'
    figure.suptitle(f'{state.title()} redundancy R_NN (tau=slice mean, {metric["key_position"]}) — {scope}', fontsize=12)
    _save_figure(figure, destination)


def _save_figure(figure, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    figure.savefig(temporary, format='png', facecolor='white')
    figure.clear()
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description='Compute per-frame K/V nearest-neighbor redundancy scores from a KV snapshot.')
    parser.add_argument('result', type=Path)
    parser.add_argument('--position', choices=['post', 'pre'], default='post', help='Key position relative to RoPE')
    parser.add_argument('--output-dir', type=Path, help='Default: uniqueness/<position>/ beside result.json')
    parser.add_argument('--epsilon', type=float, default=DEFAULT_EPSILON, help='Positive scale stabilizer outside square root')
    args = parser.parse_args()
    validate_epsilon(args.epsilon)
    result = json.loads(args.result.read_text(encoding='utf-8'))
    manifest_name = 'pre_rope_key_snapshot' if args.position == 'pre' else 'kv_snapshot'
    if manifest_name not in result:
        parser.error(f'This result has no {KEY_POSITIONS[args.position]} snapshot; regenerate observation data first')
    metric = result.get('uniqueness_pre_rope' if args.position == 'pre' else 'uniqueness')
    if (metric is None or metric.get('metric') != METRIC or metric.get('key_position') != KEY_POSITIONS[args.position]
            or metric.get('epsilon') != args.epsilon):
        directory = args.result.parent / ('key_pre_rope' if args.position == 'pre' else 'kv')
        metric = score_snapshot(directory, result[manifest_name], epsilon=args.epsilon,
                                on_layer=lambda n, total: print(f'K/V redundancy layer {n}/{total}', flush=True))
    destination = args.output_dir or args.result.parent / 'uniqueness' / args.position
    destination.mkdir(parents=True, exist_ok=True)
    payload = dict(uniqueness=metric, key_position=KEY_POSITIONS[args.position], frames=result['frames'], model=result['model'],
                   job_id=result.get('job_id'), source_result=str(args.result.resolve()))
    temporary = destination / 'scores.tmp'
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(destination / 'scores.json')
    (destination / 'scores.csv').write_text(score_csv(metric, result['frames']), encoding='utf-8')
    for state in ('key', 'value'):
        render_scores(metric, result['frames'], destination / f'distribution-{state}.png', state=state)
    print(f'Saved K/V redundancy scores, CSV and plot to {destination}', flush=True)


if __name__ == '__main__':
    main()
