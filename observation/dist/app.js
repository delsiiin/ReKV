'use strict';
const $ = id => document.getElementById(id);
const state = {samples: [], record: null, result: null, job: null, generation: 0, selected: 0, selectedToken: 0, config: null};
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const fmt = (value, digits = 3) => value == null ? '—' : Number(value).toFixed(digits);
const pct = value => value == null ? '—' : `${fmt(value * 100, 2)}%`;
const time = value => `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, '0')}`;
const node = (tag, cls, text) => { const el = document.createElement(tag); if (cls) el.className = cls; if (text != null) el.textContent = text; return el; };

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function status(message, fraction = 0, error = false) {
  $('statusText').textContent = message;
  $('statusPercent').textContent = fraction > 0 ? `${Math.round(fraction * 100)}%` : '';
  $('progressBar').style.width = `${fraction * 100}%`;
  $('status').classList.toggle('error', error);
}
function showError(error) { status(error.message || String(error), 0, true); }
async function cancelPrevious() {
  const previous = state.job;
  state.job = null;
  if (previous) {
    try { await api(`/api/jobs/${previous}/cancel`, {}); }
    catch (error) { console.warn('Could not cancel previous observation', error); }
  }
}
function clearResults() {
  state.result = null;
  state.selected = 0;
  state.selectedToken = 0;
  $('results').hidden = true;
  $('emptyState').hidden = false;
  ['chartPanel', 'detailPanel', 'tokensPanel', 'inputDetails', 'answerPanel', 'massPanel', 'exportJson'].forEach(id => $(id).hidden = true);
  $('layer').disabled = true;
  $('exportCsv').disabled = true;
  $('cancel').hidden = true;
  $('run').disabled = false;
  $('runtime').textContent = '';
}
function fillSamples() {
  $('sample').replaceChildren();
  state.samples.filter(sample => sample.split === $('split').value).forEach((sample, index) => {
    const option = node('option', '', `${String(index + 1).padStart(2, '0')} · ${sample.video_id}`);
    option.value = sample.sample_id;
    $('sample').append(option);
  });
}
function showQuestion() {
  const question = state.record?.conversations[Number($('question').value)];
  $('questionText').textContent = question?.question || '—';
  $('answerText').textContent = question ? `参考答案：${question.answer}\n标注时间段：${question.start_time}–${question.end_time} 秒` : '';
}
async function loadSample(auto = true) {
  const generation = ++state.generation;
  clearResults();
  state.record = null;
  $('run').disabled = true;
  await cancelPrevious();
  if (generation !== state.generation) return;
  try {
    const record = await api(`/api/samples/${encodeURIComponent($('sample').value)}`);
    if (generation !== state.generation) return;
    state.record = record;
    $('sampleTitle').textContent = record.video_id;
    $('sampleTag').textContent = `RVS ${record.split.toUpperCase()}`;
    $('sampleMeta').textContent = `标注时长 ${time(record.duration || 0)} · ${record.conversations.length} 个问题 · ${record.video_path}`;
    $('rawSample').textContent = JSON.stringify(record, null, 2);
    $('question').replaceChildren();
    record.conversations.forEach((question, index) => {
      const option = node('option', '', `${index + 1}. [${time(question.start_time)}–${time(question.end_time)}] ${question.question}`);
      option.value = index;
      $('question').append(option);
    });
    showQuestion();
    $('run').disabled = false;
    status('样本已就绪。生成数据后，将先显示采样帧，再更新注意力统计。');
    if (auto && $('autoRun').checked) await run();
  } catch (error) { if (generation === state.generation) showError(error); }
}
function parameters() {
  return {sample_id: $('sample').value, max_frames: Number($('maxFrames').value),
    sample_fps: Number($('fps').value), start_time: Number($('startTime').value),
    mode: $('mode').value, question_index: Number($('question').value), max_new_tokens: Number($('maxAnswerTokens').value)};
}
async function run() {
  if (!state.record || !$('controls').reportValidity()) return;
  const generation = ++state.generation;
  const params = parameters();
  clearResults();
  $('run').disabled = true;
  status('正在创建观察任务…');
  await cancelPrevious();
  if (generation !== state.generation) return;
  try {
    let job = await api('/api/jobs', params);
    if (generation !== state.generation) {
      await api(`/api/jobs/${job.id}/cancel`, {});
      return;
    }
    state.job = job.id;
    $('cancel').hidden = false;
    let previewShown = false;
    while (generation === state.generation) {
      status(job.message, job.progress || 0, job.status === 'failed');
      if (job.preview && !previewShown) {
        previewShown = true;
        $('results').hidden = false;
        $('emptyState').hidden = true;
        $('resultLabel').textContent = '采样帧已解码 · 等待注意力结果';
        renderSummary(job.preview.sampling);
        renderFrames(job.preview.frames, null);
      }
      if (job.status === 'complete') {
        const result = await api(job.result_url);
        if (generation !== state.generation) return;
        state.result = result;
        renderResult(job.result_url, job.cached);
        status(job.cached ? '已读取相同输入的本地缓存。' : '观察计算完成。可切换层、查看帧与 token 分数。', 1);
        break;
      }
      if (job.status === 'failed' || job.status === 'cancelled') break;
      await sleep(1000);
      if (generation !== state.generation) return;
      job = await api(`/api/jobs/${job.id}`);
    }
  } catch (error) { if (generation === state.generation) showError(error); }
  finally {
    if (generation === state.generation) {
      state.job = null;
      $('cancel').hidden = true;
      $('run').disabled = false;
    }
  }
}
function currentView() { return state.result?.views.find(view => view.id === $('layer').value); }
function metricCard(label, value, caption) {
  const card = node('div', 'metric-card');
  card.append(node('div', 'muted', label), node('div', 'value', value), node('div', 'caption', caption));
  return card;
}
function renderSummary(sampling) {
  const result = state.result, view = currentView();
  $('summary').replaceChildren(
    metricCard('实际处理帧', sampling.processed_frames, `${fmt(sampling.first_timestamp, 1)}–${fmt(sampling.last_timestamp, 1)} 秒 · ${sampling.truncated ? '已限制帧数' : '已到视频末尾'}`),
    metricCard('历史 / 观察 token', result ? `${result.sequence.prefix_tokens} / 32` : '—', result ? `每帧 ${result.sequence.tokens_per_frame} 个 token` : '正在提取视频特征'),
    metricCard('历史视频 attention mass', view ? pct(view.video_mass) : '—', view ? `历史文本 ${pct(view.text_mass)} · 窗口 ${pct(view.window_mass)}` : '完整因果注意力'),
    metricCard('帧内归一化熵 / 均值', view ? fmt(mean(view.frames.map(frame => frame.normalized_entropy))) : '—', '0 集中 ← 分布 → 均匀 1')
  );
}
function mean(values) { const good = values.filter(value => value != null); return good.length ? good.reduce((a, b) => a + b, 0) / good.length : null; }
function renderResult(url, cached) {
  const result = state.result;
  $('results').hidden = false;
  $('emptyState').hidden = true;
  ['chartPanel', 'detailPanel', 'tokensPanel', 'inputDetails', 'answerPanel', 'massPanel'].forEach(id => $(id).hidden = false);
  $('inputMetadata').textContent = JSON.stringify({parameters: result.parameters, sampling: result.sampling, model: result.model,
    token_layout: {prompt_tokens: result.sequence.prompt_tokens, video_tokens: result.sequence.video_tokens,
      total_tokens: result.sequence.total_tokens, window: [result.sequence.window_start, result.sequence.window_end]}, method: result.method,
    spatial: {...result.spatial, cells: `${result.spatial.cells.length} cells; full coordinates in JSON`}, answer: result.answer}, null, 2);
  $('layer').replaceChildren(node('option', '', '全部层平均'));
  $('layer').options[0].value = 'all';
  result.views.slice(1).forEach(view => {
    const option = node('option', '', `第 ${Number(view.id) + 1} 层`); option.value = view.id; $('layer').append(option);
  });
  $('layer').disabled = false;
  $('exportJson').hidden = false;
  $('exportJson').href = url;
  $('exportCsv').disabled = false;
  $('resultLabel').textContent = `${result.model.layers} 层 · ${result.model.heads} heads · ${cached ? '本地缓存' : `${fmt(result.total_elapsed_seconds, 1)} 秒`}`;
  $('runtime').textContent = `torch ${result.model.versions.torch} / transformers ${result.model.versions.transformers}`;
  renderAnswer();
  renderView();
}
function renderAnswer() {
  const result = state.result, answer = result.answer;
  $('answerQuestion').textContent = answer.question;
  $('modelAnswer').textContent = answer.text || '（模型输出了终止符，没有生成可显示的文字。）';
  $('referenceAnswer').textContent = result.sample.conversations[answer.question_index].answer ?? '未提供';
  $('answerStatus').textContent = {eos: '已完成', max_new_tokens: '达到回答长度上限', context_limit: '达到上下文上限'}[answer.finish_reason] || answer.finish_reason;
  $('answerMeta').textContent = `输入 ${result.frames.length} 帧 / ${fmt(result.sampling.first_timestamp, 2)}–${fmt(result.sampling.last_timestamp, 2)} 秒 · 贪心解码 ${answer.generated_tokens} token · ${fmt(answer.elapsed_seconds, 2)} 秒。回答仅使用这些帧和问题；参考答案未输入模型。`;
}
function renderView() {
  const result = state.result, view = currentView();
  if (!result || !view) return;
  renderSummary(result.sampling);
  renderFrames(result.frames, view);
  renderCharts();
  renderMassDistribution();
  renderDetail();
  renderTokens();
  const text = $('textTokens'); text.replaceChildren();
  result.sequence.text_tokens.forEach(token => {
    const item = node('div', 'text-token', token.text);
    item.append(node('small', '', `#${token.position} · ${token.position < view.scores.length ? view.scores[token.position].toExponential(4) : '观察窗口'}`));
    text.append(item);
  });
}
function renderFrames(frames, view) {
  $('frameCount').textContent = frames.length;
  const container = $('frames'); container.replaceChildren();
  frames.forEach((frame, index) => {
    const card = node('button', `frame-card${index === state.selected && view ? ' selected' : ''}`);
    card.type = 'button'; card.dataset.index = index;
    card.setAttribute('aria-label', `查看第 ${index + 1} 帧，${fmt(frame.timestamp, 2)} 秒`);
    const img = node('img'); img.src = frame.image_url; img.alt = `实际采样帧 ${index + 1}，源帧 ${frame.source_index}`; img.loading = 'lazy';
    const caption = node('div', 'frame-caption'), top = node('div', 'top');
    top.append(node('span', '', `FRAME ${String(index + 1).padStart(2, '0')}`), node('span', '', `${fmt(frame.timestamp, 2)}s`));
    caption.append(top, node('div', 'source', `源帧 #${frame.source_index} · ${frame.width} × ${frame.height}`));
    const stats = view?.frames[index];
    caption.append(node('div', 'stats', stats ? `M ${pct(stats.mass)} · H̄ ${fmt(stats.normalized_entropy)}${stats.window_tokens ? ' · 部分为窗口' : ''}` : '等待注意力统计…'));
    card.append(img, caption);
    card.addEventListener('click', () => { if (state.result) selectFrame(index, true); });
    container.append(card);
  });
}
function selectFrame(index, scroll = false, tokenIndex = null) {
  state.selected = index;
  state.selectedToken = tokenIndex ?? 0;
  document.querySelectorAll('.frame-card').forEach(card => card.classList.toggle('selected', Number(card.dataset.index) === index));
  renderCharts(); renderMassDistribution(); renderDetail();
  if (scroll) $('detailPanel').scrollIntoView({behavior: 'smooth', block: 'nearest'});
}
function svgNode(tag, attributes, text) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  Object.entries(attributes).forEach(([key, value]) => el.setAttribute(key, value));
  if (text != null) el.textContent = text;
  return el;
}
function renderCharts() {
  const view = currentView(); if (!view) return;
  const frames = state.result.frames, n = frames.length, width = 900;
  const svg = svgNode('svg', {viewBox: `0 0 ${width} 295`, role: 'img', 'aria-label': '逐帧 attention mass 与归一化条件熵'});
  const x = i => n === 1 ? 470 : 65 + i / (n - 1) * 790;
  [{key: 'mass', top: 34, bottom: 111, color: '#087f80', label: 'ATTENTION MASS', max: Math.max(...view.frames.map(f => f.mass), 0.000001)},
   {key: 'normalized_entropy', top: 167, bottom: 244, color: '#d77a3a', label: 'NORMALIZED CONDITIONAL ENTROPY', max: 1}].forEach(chart => {
    svg.append(svgNode('text', {x: 65, y: chart.top - 16}, chart.label));
    for (const fraction of [0, .5, 1]) {
      const y = chart.bottom - fraction * (chart.bottom - chart.top);
      svg.append(svgNode('line', {x1: 65, x2: 855, y1: y, y2: y, stroke: '#e6ecef', 'stroke-dasharray': '3 4'}));
      svg.append(svgNode('text', {x: 55, y: y + 4, 'text-anchor': 'end'}, chart.key === 'mass' ? pct(fraction * chart.max) : fmt(fraction, 1)));
    }
    let path = '', connected = false;
    view.frames.forEach((frame, index) => {
      const value = frame[chart.key];
      if (value == null) { connected = false; return; }
      const y = chart.bottom - value / chart.max * (chart.bottom - chart.top);
      path += `${connected ? 'L' : 'M'}${x(index)},${y} `; connected = true;
    });
    svg.append(svgNode('path', {d: path, fill: 'none', stroke: chart.color, 'stroke-width': 2}));
    svg.append(svgNode('line', {x1: x(state.selected), x2: x(state.selected), y1: chart.top, y2: chart.bottom, stroke: '#99b1ba', 'stroke-dasharray': '2 4'}));
    view.frames.forEach((frame, index) => {
      const value = frame[chart.key]; if (value == null) return;
      const circle = svgNode('circle', {cx: x(index), cy: chart.bottom - value / chart.max * (chart.bottom - chart.top), r: index === state.selected ? 5.5 : 3.5, fill: chart.color, tabindex: 0, role: 'button', 'aria-label': `帧 ${index + 1} ${chart.label} ${fmt(value, 5)}`});
      circle.append(svgNode('title', {}, `帧 ${index + 1} · ${fmt(frames[index].timestamp, 2)}s · ${fmt(value, 6)}`));
      circle.addEventListener('click', () => selectFrame(index));
      circle.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectFrame(index); } });
      svg.append(circle);
    });
  });
  frames.forEach((frame, index) => {
    if (index % Math.max(1, Math.ceil(n / 7)) === 0 || index === n - 1)
      svg.append(svgNode('text', {x: x(index), y: 272, 'text-anchor': 'middle'}, `${fmt(frame.timestamp, 1)}s`));
  });
  $('charts').replaceChildren(svg);
}
function renderMassDistribution() {
  const view = currentView(); if (!view) return;
  const groups = [['历史文本', view.text_mass, '#75889b'], ['历史视频', view.video_mass, '#087f80'], ['观察窗口', view.window_mass, '#d9a15f']];
  $('massComposition').replaceChildren(); $('massLegend').replaceChildren();
  for (const [label, value, fill] of groups) {
    const segment = node('div', 'mass-segment', value > .075 ? pct(value) : '');
    segment.style.width = `${value * 100}%`; segment.style.background = fill; segment.title = `${label}：${pct(value)}`;
    $('massComposition').append(segment);
    const legend = node('div', 'mass-legend-item');
    const swatch = node('span', 'mass-swatch'); swatch.style.background = fill;
    legend.append(swatch, node('span', '', label), node('b', '', pct(value))); $('massLegend').append(legend);
  }
  $('massComposition').setAttribute('aria-label', groups.map(([label, value]) => `${label} ${pct(value)}`).join('，'));
  const relative = $('massScale').value === 'video';
  const values = view.frames.map(frame => relative ? (view.video_mass > 0 ? frame.mass / view.video_mass : 0) : frame.mass);
  const max = Math.max(...values, 1e-12), n = values.length;
  const svg = svgNode('svg', {viewBox: '0 0 900 225', role: 'img', 'aria-label': relative ? '每帧占历史视频 attention mass 的比例' : '每帧占总 attention mass 的比例'});
  const left = 65, span = 790 / n, base = 177, height = 135;
  for (const fraction of [0, .5, 1]) {
    const y = base - fraction * height;
    svg.append(svgNode('line', {x1: left, x2: 855, y1: y, y2: y, stroke: '#e6ecef', 'stroke-dasharray': '3 4'}));
    svg.append(svgNode('text', {x: 55, y: y + 4, 'text-anchor': 'end'}, pct(max * fraction)));
  }
  values.forEach((value, index) => {
    const x = left + span * index + span * .15, y = base - value / max * height;
    const bar = svgNode('rect', {x, y, width: span * .7, height: Math.max(1, base - y), rx: Math.min(3, span * .1),
      fill: index === state.selected ? '#d77a3a' : '#087f80', tabindex: 0, role: 'button', 'aria-label': `帧 ${index + 1} ${pct(value)}`});
    bar.append(svgNode('title', {}, `帧 ${index + 1} / ${fmt(state.result.frames[index].timestamp, 2)}s / ${pct(value)}`));
    bar.addEventListener('click', () => selectFrame(index, true));
    bar.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectFrame(index, true); } });
    svg.append(bar);
    if (n <= 12) svg.append(svgNode('text', {x: left + span * (index + .5), y: y - 8, 'text-anchor': 'middle'}, pct(value)));
    if (index % Math.max(1, Math.ceil(n / 12)) === 0 || index === n - 1)
      svg.append(svgNode('text', {x: left + span * (index + .5), y: 199, 'text-anchor': 'middle'}, `F${index + 1}`));
  });
  $('massBars').replaceChildren(svg);
  $('massCaption').textContent = relative ? '每柱 = M_f / 历史视频总 mass；有视频注意力时，所有柱相加为 100%。点击柱形定位帧。'
    : `每柱为该帧的原始 attention mass；所有柱相加为 ${pct(view.video_mass)}。窗口中的视频 token 不计入历史帧 mass。`;
}
function color(ratio) { const t = Math.max(0, Math.min(1, ratio)); return `rgb(${Math.round(232 - t * 224)},${Math.round(244 - t * 117)},${Math.round(242 - t * 114)})`; }
function renderDetail() {
  const view = currentView(); if (!view) return;
  const frame = state.result.frames[state.selected], stats = view.frames[state.selected];
  $('detailTitle').textContent = `FRAME ${String(state.selected + 1).padStart(2, '0')} · ${fmt(frame.timestamp, 2)} 秒`;
  $('detailRange').textContent = `token [${frame.token_start}, ${frame.token_end}) · 源帧 #${frame.source_index}`;
  $('detailImage').src = frame.image_url;
  $('detailImage').alt = `第 ${state.selected + 1} 张实际采样帧，源帧 ${frame.source_index}`;
  $('imageStage').style.aspectRatio = `${frame.width} / ${frame.height}`;
  $('frameMetrics').replaceChildren(...[
    ['Attention mass', pct(stats.mass)], ['帧内条件熵 / nats', fmt(stats.entropy, 4)],
    ['归一化熵', fmt(stats.normalized_entropy, 4)], ['有效 token 数 exp(H)', fmt(stats.effective_tokens, 1)]
  ].map(([label, value]) => { const el = node('div', 'metric'); el.append(node('div', 'muted', label), node('b', '', value)); return el; }));
  renderImageOverlay();
  renderHeatmap();
  activateToken(state.selectedToken);
}
function renderHeatmap() {
  const view = currentView(); if (!view) return;
  const frame = state.result.frames[state.selected], stats = view.frames[state.selected];
  const canvas = $('heatmap'), ctx = canvas.getContext('2d');
  const total = frame.token_end - frame.token_start;
  const columns = state.result.sequence.grid_size || Math.ceil(Math.sqrt(total));
  const rows = Math.ceil(total / columns), cell = canvas.width / columns;
  canvas.height = rows * cell;
  const scores = view.scores.slice(frame.token_start, frame.token_end), max = Math.max(...scores, 1e-30);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < total; i++) {
    ctx.fillStyle = i < scores.length ? color(scores[i] / max) : '#c5ced3';
    ctx.fillRect((i % columns) * cell + 1, Math.floor(i / columns) * cell + 1, cell - 2, cell - 2);
  }
  ctx.strokeStyle = '#ed762c'; ctx.lineWidth = 3;
  ctx.strokeRect((state.selectedToken % columns) * cell + 1.5, Math.floor(state.selectedToken / columns) * cell + 1.5, cell - 3, cell - 3);
  $('heatmapHint').textContent = `${stats.scored_tokens} 个历史 token · ${stats.window_tokens} 个窗口 token`;
  canvas.onmousemove = event => {
    const rect = canvas.getBoundingClientRect();
    const index = Math.floor((event.clientY - rect.top) / rect.height * rows) * columns + Math.floor((event.clientX - rect.left) / rect.width * columns);
    if (index >= total || index < 0) return;
    activateToken(index);
  };
}
function boxAttributes(box, frame) {
  return {x: box[0] * frame.width, y: box[1] * frame.height,
    width: (box[2] - box[0]) * frame.width, height: (box[3] - box[1]) * frame.height};
}
function renderImageOverlay() {
  const view = currentView(); if (!view) return;
  const frame = state.result.frames[state.selected], stats = view.frames[state.selected], layout = state.result.spatial;
  const svg = $('imageOverlay'); svg.setAttribute('viewBox', `0 0 ${frame.width} ${frame.height}`); svg.replaceChildren();
  const frameScores = view.scores.slice(frame.token_start, frame.token_end);
  const conditional = $('overlayMode').value === 'conditional';
  let max = 1e-30;
  if (conditional) max = Math.max(...frameScores, max);
  else for (const f of state.result.frames) for (let i = f.token_start; i < Math.min(f.token_end, view.scores.length); i++) max = Math.max(max, view.scores[i]);
  const enabled = $('showOverlay').checked, opacity = Number($('overlayOpacity').value);
  layout.cells.forEach(cell => {
    const score = frameScores[cell.index], missing = score === undefined;
    const rect = svgNode('rect', {...boxAttributes(cell.box, frame), fill: missing ? '#85959e' : color(score / max),
      'fill-opacity': enabled ? opacity : 0, stroke: '#ffffff', 'stroke-opacity': enabled ? .35 : 0,
      'stroke-width': .5, 'vector-effect': 'non-scaling-stroke', class: 'image-token', 'data-token': frame.token_start + cell.index});
    const probability = !missing && stats.mass > 0 ? score / stats.mass : null;
    rect.append(svgNode('title', {}, `token #${frame.token_start + cell.index} · 行 ${cell.row + 1} 列 ${cell.col + 1} · ${missing ? '观察窗口' : `mass ${pct(score)} / 帧内 p ${pct(probability)}`}`));
    rect.addEventListener('pointerenter', () => activateToken(cell.index));
    rect.addEventListener('click', () => activateToken(cell.index));
    svg.append(rect);
  });
  for (const [id, dash] of [['poolingSupport', '5 3'], ['tokenAnchor', '']]) {
    svg.append(svgNode('rect', {id, fill: 'none', stroke: '#f1843d', 'stroke-width': 2,
      'stroke-dasharray': dash, 'vector-effect': 'non-scaling-stroke', 'pointer-events': 'none', opacity: enabled ? 1 : 0}));
  }
}
function activateToken(index) {
  const view = currentView(); if (!view) return;
  const cell = state.result.spatial.cells[index]; if (!cell) return;
  state.selectedToken = index;
  const frame = state.result.frames[state.selected], stats = view.frames[state.selected], position = frame.token_start + index;
  const score = view.scores[position];
  for (const [id, box] of [['poolingSupport', cell.support_box], ['tokenAnchor', cell.box]]) {
    const el = $(id); if (el) Object.entries(boxAttributes(box, frame)).forEach(([key, value]) => el.setAttribute(key, value));
  }
  renderHeatmap();
  const bounds = cell.box.map((value, i) => fmt(value * (i % 2 ? frame.height : frame.width), 1));
  const detail = score === undefined ? '观察窗口 / 不计入历史重要性' : `原始 mass = ${score.toExponential(5)} (${fmt(score * 100, 5)}%) · 帧内 p = ${stats.mass > 0 ? fmt(score / stats.mass, 6) : '未定义'}`;
  $('spatialReadout').textContent = `token #${position} · 帧内 #${index} · 行 ${cell.row + 1} / 列 ${cell.col + 1} · 原图区域 [${bounds.join(', ')}] px\n${detail}`;
  $('heatmapHint').textContent = `#${position} · 行 ${cell.row + 1} / 列 ${cell.col + 1}${score === undefined ? ' · 观察窗口' : ''}`;
}
function tokenDescription(index, view) {
  const frame = state.result.frames.find(f => index >= f.token_start && index < f.token_end);
  const text = state.result.sequence.text_tokens.find(t => t.position === index);
  const type = frame ? `帧 ${frame.index + 1} / 帧内 token ${index - frame.token_start}` : `${text?.kind || '文本'} ${JSON.stringify(text?.text || '')}`;
  return `位置 #${index} · ${type} · ${index < view.scores.length ? `重要性 ${view.scores[index].toExponential(6)}` : '观察窗口 / 不计入历史重要性'}`;
}
function renderTokens() {
  const result = state.result, view = currentView(); if (!view) return;
  const canvas = $('tokenCanvas'), width = Math.max(200, canvas.parentElement.clientWidth - 14);
  const cell = 8, cols = Math.floor(width / cell), rows = Math.ceil(result.sequence.total_tokens / cols);
  canvas.width = cols * cell; canvas.height = rows * cell;
  const ctx = canvas.getContext('2d'), max = view.scores.reduce((a, b) => Math.max(a, b), 1e-30), log = $('logScale').checked;
  for (let i = 0; i < result.sequence.total_tokens; i++) {
    const ratio = i < view.scores.length ? view.scores[i] / max : null;
    ctx.fillStyle = ratio == null ? '#aebec6' : color(log ? Math.log1p(ratio * 1000) / Math.log(1001) : ratio);
    ctx.fillRect((i % cols) * cell, Math.floor(i / cols) * cell, cell - 1, cell - 1);
  }
  const indexAt = event => {
    const rect = canvas.getBoundingClientRect();
    return Math.floor((event.clientY - rect.top) / rect.height * canvas.height / cell) * cols + Math.floor((event.clientX - rect.left) / rect.width * canvas.width / cell);
  };
  canvas.onmousemove = event => { const i = indexAt(event); if (i < result.sequence.total_tokens) $('tokenHint').textContent = tokenDescription(i, view); };
  canvas.onclick = event => { const i = indexAt(event), frame = result.frames.find(f => i >= f.token_start && i < f.token_end); if (frame) selectFrame(frame.index, true, i - frame.token_start); };
  $('tokenHint').textContent = `按序号从左到右、逐行排列。历史范围 [0, ${result.sequence.prefix_tokens})；窗口 [${result.sequence.window_start}, ${result.sequence.window_end})。`;
}
function exportCsv() {
  const view = currentView(); if (!view) return;
  const columns = ['frame_index', 'source_index', 'timestamp', 'token_start', 'token_end', 'mass', 'entropy_nats', 'normalized_entropy', 'effective_tokens', 'scored_tokens', 'window_tokens'];
  const rows = state.result.frames.map((frame, i) => {
    const s = view.frames[i];
    return [frame.index, frame.source_index, frame.timestamp, frame.token_start, frame.token_end, s.mass, s.entropy, s.normalized_entropy, s.effective_tokens, s.scored_tokens, s.window_tokens].map(v => v == null ? '' : v).join(',');
  });
  const url = URL.createObjectURL(new Blob([columns.join(',') + '\n' + rows.join('\n')], {type: 'text/csv;charset=utf-8'}));
  const link = node('a'); link.href = url; link.download = `rvs-${state.result.sample.video_id}-layer-${view.id}.csv`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function invalidate() {
  ++state.generation; clearResults(); await cancelPrevious();
  status('输入参数已更改。点击「生成观察数据」应用新参数。');
}
$('controls').addEventListener('submit', event => { event.preventDefault(); run(); });
$('split').addEventListener('change', () => { fillSamples(); $('startTime').value = 0; loadSample(); });
$('sample').addEventListener('change', () => { $('startTime').value = 0; loadSample(); });
$('question').addEventListener('change', () => { showQuestion(); if ($('autoRun').checked) run(); else invalidate(); });
['maxFrames', 'fps', 'startTime', 'mode', 'maxAnswerTokens'].forEach(id => $(id).addEventListener('change', invalidate));
$('layer').addEventListener('change', renderView);
$('massScale').addEventListener('change', renderMassDistribution);
['showOverlay', 'overlayMode', 'overlayOpacity'].forEach(id => $(id).addEventListener('input', () => { if (state.result) { renderImageOverlay(); activateToken(state.selectedToken); } }));
$('heatmap').addEventListener('keydown', event => {
  if (!state.result) return;
  const cols = state.result.spatial.cols, moves = {ArrowLeft: -1, ArrowRight: 1, ArrowUp: -cols, ArrowDown: cols};
  if (event.key in moves) { event.preventDefault(); activateToken(Math.max(0, Math.min(state.result.spatial.cells.length - 1, state.selectedToken + moves[event.key]))); }
});
$('logScale').addEventListener('change', renderTokens);
$('exportCsv').addEventListener('click', exportCsv);
$('cancel').addEventListener('click', async () => { if (state.job) { try { const job = await api(`/api/jobs/${state.job}/cancel`, {}); status(job.message, job.progress); } catch (error) { showError(error); } } });
window.addEventListener('resize', renderTokens);
async function init() {
  try {
    [state.config, state.samples] = await Promise.all([api('/api/config'), api('/api/samples')]);
    $('modelInfo').textContent = `${state.config.model_path.split('/').pop()} · ${state.config.device}`;
    $('maxFrames').max = state.config.max_frames;
    $('maxFrames').value = Math.min(8, state.config.max_frames);
    if (!state.samples.length) throw new Error('RVS 数据集中没有样本');
    $('split').value = state.samples[0].split;
    fillSamples(); await loadSample(false);
  } catch (error) { showError(error); $('run').disabled = true; }
}
init();
