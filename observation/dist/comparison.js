'use strict';

window.comparisonViewer = (() => {
  const el = id => document.getElementById(id);
  const make = (tag, cls, text) => {
    const item = document.createElement(tag);
    if (cls) item.className = cls;
    if (text != null) item.textContent = text;
    return item;
  };
  const number = value => value == null ? '未定义' : (value === 0 ? '0' : value.toExponential(5));
  const phases = ['before', 'after', 'delta'];
  const labels = ['问题前', '问题后', '差值 · 后 − 前'];
  let result = null, comparison = null, view = null, row = null, frameValues = [], offsets = [];
  let selectedFrame = 0, selectedToken = 0, scale = 0, deltaScale = 0;

  function reset() {
    result = comparison = view = row = null; frameValues = []; offsets = [];
    selectedFrame = selectedToken = 0;
    el('comparisonPanel').hidden = true;
    el('exportComparisonCsv').disabled = true;
    el('comparisonFrames').replaceChildren();
    el('comparisonReadout').textContent = '';
    el('comparisonLegend').replaceChildren();
    el('comparisonQueries').textContent = '';
    el('comparisonFrame').replaceChildren();
    el('comparisonFrame').disabled = true;
  }
  function configure(value) {
    reset(); result = value; comparison = value.attention_comparison;
    el('comparisonPanel').hidden = false;
    for (const id of ['comparisonControls', 'comparisonReadout', 'comparisonQueryDetails', 'comparisonNote']) {
      el(id).hidden = !comparison;
    }
    if (!comparison) {
      el('comparisonStatus').textContent = '旧结果不含问题前后对照；请重新生成观察数据。';
      return;
    }
    const {before, after} = comparison.queries;
    const range = query => `[${query.positions[0]}, ${query.positions.at(-1) + 1})`;
    el('comparisonStatus').textContent =
      `问题前：视频末尾 ${before.count} 个 query ${range(before)} · 问题后：正文 ${after.count} 个 query ${range(after)}。仅展示当前选中帧，随帧与观察层切换。`;
    el('comparisonQueries').textContent = JSON.stringify(comparison.queries, null, 2);
    let offset = 0;
    result.frames.forEach((frame, index) => {
      offsets.push(offset); offset += frame.token_end - frame.token_start;
      const option = make('option', '', `FRAME ${index + 1} · ${frame.timestamp.toFixed(2)} 秒`);
      option.value = index; el('comparisonFrame').append(option);
    });
    el('comparisonFrame').disabled = false;
    const card = make('article', 'comparison-frame selected');
    const heading = make('div', 'comparison-frame-heading');
    const preview = make('div', 'comparison-preview'), image = make('img');
    preview.append(image);
    const caption = make('div'), title = make('strong'), metadata = make('div', 'muted');
    caption.append(title, metadata);
    const stats = make('div', 'comparison-frame-stats');
    heading.append(preview, caption, stats); card.append(heading);
    const triptych = make('div', 'comparison-triptych');
    const canvases = phases.map((phase, p) => {
      const column = make('div');
      const canvas = make('canvas', 'comparison-canvas');
      canvas.tabIndex = 0;
      canvas.dataset.phase = phase;
      canvas.addEventListener('mousemove', event => {
        const token = tokenAt(canvas, event);
        if (token !== null) readout(selectedFrame, token);
      });
      canvas.addEventListener('mouseleave', () => readout(selectedFrame, selectedToken));
      canvas.addEventListener('click', event => {
        const token = tokenAt(canvas, event);
        if (token !== null) window.selectFrame(selectedFrame, false, token);
      });
      canvas.addEventListener('keydown', event => {
        const moves = {ArrowLeft: -1, ArrowRight: 1, ArrowUp: -result.spatial.cols, ArrowDown: result.spatial.cols};
        if (!(event.key in moves)) return;
        event.preventDefault();
        const token = Math.max(0, Math.min(result.spatial.cells.length - 1, selectedToken + moves[event.key]));
        window.selectFrame(selectedFrame, false, token);
      });
      column.append(make('h3', '', labels[p]), canvas); triptych.append(column);
      return canvas;
    });
    card.append(triptych); el('comparisonFrames').append(card);
    row = {card, image, title, metadata, stats, canvases};
    image.addEventListener('load', () => {
      if (view && row?.image === image && el('comparisonDisplay').value === 'overlay') drawSelectedFrame();
    });
    updateSelectedFrame();
    el('exportComparisonCsv').disabled = false;
  }
  function tokenAt(canvas, event) {
    const bounds = canvas.getBoundingClientRect();
    const x = (event.clientX - bounds.left) / bounds.width;
    const y = (event.clientY - bounds.top) / bounds.height;
    if (x < 0 || x >= 1 || y < 0 || y >= 1) return null;
    if (el('comparisonDisplay').value === 'overlay') {
      return result.spatial.cells.find(cell => x >= cell.box[0] && x < cell.box[2] &&
        y >= cell.box[1] && y < cell.box[3])?.index ?? null;
    }
    const index = Math.floor(y * result.spatial.rows) * result.spatial.cols + Math.floor(x * result.spatial.cols);
    return index < result.spatial.cells.length ? index : null;
  }
  function metricValues(index, phase) {
    const frame = result.frames[index], stats = view[phase].frames[index];
    const scores = view[phase].scores.slice(offsets[index], offsets[index] + frame.token_end - frame.token_start);
    return el('comparisonMetric').value === 'mass' ? scores : scores.map(score => stats.mass > 0 ? score / stats.mass : null);
  }
  function render(layer) {
    if (!comparison) return;
    view = comparison.views.find(item => item.id === (layer ?? view?.id ?? 'all'));
    if (!view) return;
    scale = deltaScale = 0;
    frameValues = result.frames.map((frame, index) => {
      const before = metricValues(index, 'before'), after = metricValues(index, 'after');
      const delta = before.map((value, i) => value == null || after[i] == null ? null : after[i] - value);
      for (const values of [before, after]) for (const value of values) if (value != null) scale = Math.max(scale, value);
      for (const value of delta) if (value != null) deltaScale = Math.max(deltaScale, Math.abs(value));
      return {before, after, delta};
    });
    legend();
    updateSelectedFrame();
    drawSelectedFrame();
    readout(selectedFrame, selectedToken);
  }
  function updateSelectedFrame() {
    if (!row) return;
    const frame = result.frames[selectedFrame];
    el('comparisonFrame').value = selectedFrame;
    row.card.dataset.frame = selectedFrame;
    row.title.textContent = `FRAME ${String(selectedFrame + 1).padStart(2, '0')} · ${frame.timestamp.toFixed(2)} 秒`;
    row.metadata.textContent = `源帧 #${frame.source_index} · token [${frame.token_start}, ${frame.token_end})`;
    row.image.alt = `第 ${selectedFrame + 1} 帧原图`;
    if (row.image.getAttribute('src') !== frame.image_url) row.image.src = frame.image_url;
    row.canvases.forEach((canvas, p) =>
      canvas.setAttribute('aria-label', `第 ${selectedFrame + 1} 帧 ${labels[p]} token 热图，方向键选择 token`));
    if (view) {
      const a = view.before.frames[selectedFrame], b = view.after.frames[selectedFrame];
      row.stats.textContent = `帧 mass ${number(a.mass)} → ${number(b.mass)}\n条件熵 ${number(a.entropy)} → ${number(b.entropy)} nats`;
    }
  }
  function legend() {
    const unit = el('comparisonMetric').value === 'mass' ? '原始 mass' : '帧内概率';
    const absolute = make('div', 'comparison-scale');
    absolute.append(make('span', '', `${unit} · 前后 / 全部帧共用标尺`), make('span', 'comparison-ramp'),
      make('span', '', `0 → ${number(scale)}`));
    const difference = make('div', 'comparison-scale');
    difference.append(make('span', '', '差值 · 蓝色减少 / 橙色增加'), make('span', 'comparison-ramp diverging'),
      make('span', '', `${number(-deltaScale)} → 0 → ${number(deltaScale)}`));
    el('comparisonLegend').replaceChildren(absolute, difference, make('span', 'muted', '灰色：未定义'));
  }
  function color(value, difference) {
    if (value == null) return '#bbc6ce';
    if (!difference) {
      const t = scale > 0 ? Math.min(1, Math.max(0, value / scale)) : 0;
      return `rgb(${Math.round(232 - t * 224)},${Math.round(244 - t * 117)},${Math.round(242 - t * 114)})`;
    }
    const t = deltaScale > 0 ? Math.min(1, Math.abs(value / deltaScale)) : 0;
    const target = value < 0 ? [42, 105, 175] : [205, 91, 36], middle = [245, 245, 242];
    return `rgb(${middle.map((v, i) => Math.round(v + (target[i] - v) * t)).join(',')})`;
  }
  function box(cell, width, height, overlay) {
    return overlay ? [cell.box[0] * width, cell.box[1] * height,
      (cell.box[2] - cell.box[0]) * width, (cell.box[3] - cell.box[1]) * height] :
      [cell.col * width / result.spatial.cols, cell.row * height / result.spatial.rows,
        width / result.spatial.cols, height / result.spatial.rows];
  }
  function drawSelectedFrame() {
    const values = frameValues[selectedFrame];
    if (!row || !values) return;
    const overlay = el('comparisonDisplay').value === 'overlay', frame = result.frames[selectedFrame];
    row.canvases.forEach((canvas, p) => {
      canvas.width = overlay ? 420 : 280;
      canvas.height = Math.round(canvas.width * (overlay ? frame.height / frame.width : result.spatial.rows / result.spatial.cols));
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#edf1f3'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (overlay && row.image.complete && row.image.naturalWidth) {
        ctx.drawImage(row.image, 0, 0, canvas.width, canvas.height);
      }
      ctx.globalAlpha = overlay ? Number(el('comparisonOpacity').value) : 1;
      for (const cell of result.spatial.cells) {
        ctx.fillStyle = color(values[phases[p]][cell.index], p === 2);
        const [x, y, width, height] = box(cell, canvas.width, canvas.height, overlay);
        const gap = overlay ? 0 : 1;
        ctx.fillRect(x + gap, y + gap, width - gap * 2, height - gap * 2);
      }
      ctx.globalAlpha = 1;
      const cell = result.spatial.cells[selectedToken];
      if (cell) {
        const [x, y, width, height] = box(cell, canvas.width, canvas.height, overlay);
        ctx.strokeStyle = '#ffb000'; ctx.lineWidth = 3;
        ctx.strokeRect(x + 1.5, y + 1.5, Math.max(0, width - 3), Math.max(0, height - 3));
      }
    });
  }
  function highlightToken(frame, token) {
    if (!comparison) return;
    const previous = selectedFrame;
    selectedFrame = frame; selectedToken = token;
    if (previous !== frame) updateSelectedFrame();
    drawSelectedFrame(); readout(frame, token);
  }
  function readout(index, token) {
    if (!view || !frameValues[index]) return;
    const frame = result.frames[index], cell = result.spatial.cells[token];
    if (!cell) return;
    const position = offsets[index] + token, before = view.before.scores[position], after = view.after.scores[position];
    const bm = view.before.frames[index].mass, am = view.after.frames[index].mass;
    const bp = bm > 0 ? before / bm : null, ap = am > 0 ? after / am : null;
    const bounds = cell.box.map((value, i) => (value * (i % 2 ? frame.height : frame.width)).toFixed(1));
    el('comparisonReadout').textContent =
      `FRAME ${index + 1} · token #${comparison.token_positions[position]} · 帧内 #${token} · 行 ${cell.row + 1} / 列 ${cell.col + 1} · 原图 [${bounds.join(', ')}] px\n` +
      `mass：前 ${number(before)} · 后 ${number(after)} · 差值 ${number(after - before)}\n` +
      `帧内 p：前 ${number(bp)} · 后 ${number(ap)} · 差值 ${number(bp == null || ap == null ? null : ap - bp)}`;
  }
  function exportCsv() {
    if (!view) return;
    const columns = ['layer', 'video_window_size', 'before_query_count', 'after_query_count',
      'frame_index', 'source_index', 'timestamp', 'token_position', 'frame_token', 'row', 'col',
      'before_mass', 'after_mass', 'delta_mass', 'before_probability', 'after_probability', 'delta_probability'];
    const lines = [columns.join(',')];
    result.frames.forEach((frame, index) => {
      const bm = view.before.frames[index].mass, am = view.after.frames[index].mass;
      result.spatial.cells.forEach(cell => {
        const position = offsets[index] + cell.index;
        const before = view.before.scores[position], after = view.after.scores[position];
        const bp = bm > 0 ? before / bm : null, ap = am > 0 ? after / am : null;
        lines.push([view.id, comparison.queries.before.count, comparison.queries.before.count, comparison.queries.after.count,
          frame.index, frame.source_index, frame.timestamp, comparison.token_positions[position], cell.index, cell.row, cell.col,
          before, after, after - before, bp, ap, bp == null || ap == null ? null : ap - bp].map(v => v ?? '').join(','));
      });
    });
    const url = URL.createObjectURL(new Blob([lines.join('\n') + '\n'], {type: 'text/csv;charset=utf-8'}));
    const link = make('a'); link.href = url;
    link.download = `rvs-${result.sample.video_id}-comparison-N${comparison.queries.before.count}-layer-${view.id}.csv`;
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  for (const id of ['comparisonMetric', 'comparisonDisplay', 'comparisonOpacity']) {
    el(id).addEventListener('input', () => render());
  }
  el('exportComparisonCsv').addEventListener('click', exportCsv);
  el('comparisonFrame').addEventListener('change', () => {
    if (comparison) window.selectFrame(Number(el('comparisonFrame').value));
  });
  return {reset, configure, render, highlightToken};
})();
