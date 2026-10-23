'use strict';
const $ = id => document.getElementById(id);
const state = {samples: [], record: null, result: null, job: null, generation: 0, selected: 0, selectedToken: 0, config: null};
const tokenSimilarity = {controller: null, request: 0, data: null, col: 0};
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
  document.dispatchEvent(new Event('observation:clear'));
  tokenSimilarity.controller?.abort(); tokenSimilarity.request++; tokenSimilarity.data = null;
  $('exportSegmentCsv').disabled = true;
  state.result = null;
  state.selected = 0;
  state.selectedToken = 0;
  $('results').hidden = true;
  $('emptyState').hidden = false;
  ['chartPanel', 'detailPanel', 'tokensPanel', 'inputDetails', 'answerPanel', 'massPanel', 'exportJson', 'segmentPanel', 'segmentSimilarityPanel'].forEach(id => $(id).hidden = true);
  $('layer').disabled = true;
  $('similarityLayer').disabled = true;
  $('similarityRepresentation').disabled = true;
  $('exportCsv').disabled = true;
  $('cancel').hidden = true;
  $('run').disabled = false;
  $('runtime').textContent = '';
  updateWindowLabels();
}
function updateWindowLabels() {
  const n = $('videoWindowSize').value || 'N';
  $('mode').options[0].textContent = `视频编码末尾 ${n} token`;
  const active = state.result?.sequence.window_size ?? ($('mode').value === 'video' ? n : 32);
  $('windowBadge').textContent = `${active} TOKEN WINDOW`;
  $('methodWindowSize').textContent = active;
  $('windowTokenCaption').textContent = `包含提示词和所有视频 token；最后 ${active} 个位置为灰色观察窗口。悬停读数，点击定位帧。`;
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
    mode: $('mode').value, question_index: Number($('question').value), max_new_tokens: Number($('maxAnswerTokens').value),
    video_window_size: Number($('videoWindowSize').value), segment_enabled: $('segmentEnabled').checked,
    segment_ema_alpha: Number($('emaAlpha').value),
    segment_min_frames: Number($('segmentMinFrames').value), summary_max_new_tokens: Number($('summaryMaxTokens').value)};
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
    metricCard('历史 / 观察 token', result ? `${result.sequence.prefix_tokens} / ${result.sequence.window_size}` : '—', result ? `每帧 ${result.sequence.tokens_per_frame} 个 token` : '正在提取视频特征'),
    metricCard('历史视频 attention mass', view ? pct(view.video_mass) : '—', view ? `历史文本 ${pct(view.text_mass)} · 窗口 ${pct(view.window_mass)}` : '完整因果注意力'),
  );
}
function renderResult(url, cached) {
  const result = state.result;
  $('results').hidden = false;
  $('emptyState').hidden = true;
  ['chartPanel', 'detailPanel', 'tokensPanel', 'inputDetails', 'answerPanel', 'massPanel'].forEach(id => $(id).hidden = false);
  $('inputMetadata').textContent = JSON.stringify({parameters: result.parameters, sampling: result.sampling, model: result.model,
    token_layout: {prompt_tokens: result.sequence.prompt_tokens, video_tokens: result.sequence.video_tokens,
      total_tokens: result.sequence.total_tokens, window: [result.sequence.window_start, result.sequence.window_end]}, method: result.method,
    spatial: {...result.spatial, cells: `${result.spatial.cells.length} cells; full coordinates in JSON`},
    answer: result.answer}, null, 2);
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
  renderSegments();
  renderSegmentSimilarity();
  updateWindowLabels();
  renderView();
  document.dispatchEvent(new Event('observation:result'));
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
    caption.append(node('div', 'stats', stats ? `M ${pct(stats.mass)}${stats.window_tokens ? ' · 部分为窗口' : ''}` : '等待注意力统计…'));
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
  const svg = svgNode('svg', {viewBox: `0 0 ${width} 165`, role: 'img', 'aria-label': '逐帧 attention mass'});
  const x = i => n === 1 ? 470 : 65 + i / (n - 1) * 790;
  [{key: 'mass', top: 34, bottom: 111, color: '#087f80', label: 'ATTENTION MASS', min: 0, max: Math.max(...view.frames.map(f => f.mass), 0.000001)}].forEach(chart => {
    const yFor = value => chart.bottom - (value - chart.min) / (chart.max - chart.min) * (chart.bottom - chart.top);
    const clipId = `chart-clip-${chart.key}`;
    const defs = svgNode('defs', {}), clip = svgNode('clipPath', {id: clipId});
    clip.append(svgNode('rect', {x: 59, y: chart.top, width: 802, height: chart.bottom - chart.top}));
    defs.append(clip); svg.append(defs);
    svg.append(svgNode('text', {x: 65, y: chart.top - 16}, chart.label));
    for (const fraction of [0, .5, 1]) {
      const y = chart.bottom - fraction * (chart.bottom - chart.top);
      svg.append(svgNode('line', {x1: 65, x2: 855, y1: y, y2: y, stroke: '#e6ecef', 'stroke-dasharray': '3 4'}));
      const tick = chart.min + fraction * (chart.max - chart.min);
      svg.append(svgNode('text', {x: 55, y: y + 4, 'text-anchor': 'end'}, chart.key === 'mass' ? pct(tick) : fmt(tick, 1)));
    }
    let path = '', connected = false;
    view.frames.forEach((frame, index) => {
      const value = frame[chart.key];
      if (value == null) { connected = false; return; }
      const y = yFor(value);
      path += `${connected ? 'L' : 'M'}${x(index)},${y} `; connected = true;
    });
    svg.append(svgNode('path', {d: path, fill: 'none', stroke: chart.color, 'stroke-width': 2, 'clip-path': `url(#${clipId})`}));
    svg.append(svgNode('line', {x1: x(state.selected), x2: x(state.selected), y1: chart.top, y2: chart.bottom, stroke: '#99b1ba', 'stroke-dasharray': '2 4'}));
    view.frames.forEach((frame, index) => {
      const value = frame[chart.key]; if (value == null || value < chart.min || value > chart.max) return;
      const circle = svgNode('circle', {cx: x(index), cy: yFor(value), r: index === state.selected ? 5.5 : 3.5, fill: chart.color, tabindex: 0, role: 'button', 'aria-label': `帧 ${index + 1} ${chart.label} ${fmt(value, 5)}`});
      circle.append(svgNode('title', {}, `帧 ${index + 1} · ${fmt(frames[index].timestamp, 2)}s · ${fmt(value, 6)}`));
      circle.addEventListener('click', () => selectFrame(index));
      circle.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectFrame(index); } });
      svg.append(circle);
    });
  });
  frames.forEach((frame, index) => {
    if (index % Math.max(1, Math.ceil(n / 7)) === 0 || index === n - 1)
      svg.append(svgNode('text', {x: x(index), y: 145, 'text-anchor': 'middle'}, `${fmt(frame.timestamp, 1)}s`));
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
    ['Attention mass', pct(stats.mass)]
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
  const columns = ['frame_index', 'source_index', 'timestamp', 'token_start', 'token_end', 'mass', 'scored_tokens', 'window_tokens'];
  const rows = state.result.frames.map((frame, i) => {
    const s = view.frames[i];
    return [frame.index, frame.source_index, frame.timestamp, frame.token_start, frame.token_end, s.mass, s.scored_tokens, s.window_tokens].map(v => v == null ? '' : v).join(',');
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
['maxFrames', 'fps', 'startTime', 'mode', 'maxAnswerTokens', 'videoWindowSize', 'segmentEnabled', 'emaAlpha', 'segmentMinFrames', 'summaryMaxTokens'].forEach(id => $(id).addEventListener('change', invalidate));
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
function renderSegments() {
  const experiment = state.result?.segment_experiment;
  $('segmentPanel').hidden = !experiment;
  $('segmentSimilarityPanel').hidden = !experiment;
  $('similarityLayer').disabled = !experiment;
  $('similarityRepresentation').disabled = !experiment;
  if (!experiment) return;
  const representations = experiment.token_similarity?.representations || ['key'];
  $('similarityRepresentation').replaceChildren();
  for (const kind of representations) {
    const option = node('option', '', kind === 'hidden' ? 'Hidden states' : 'Key states');
    option.value = kind; $('similarityRepresentation').append(option);
  }
  $('similarityLayer').replaceChildren(node('option', '', '全部层平均'));
  $('similarityLayer').options[0].value = 'all';
  for (let layer = 0; layer < state.result.model.layers; layer++) {
    const option = node('option', '', `第 ${layer + 1} 层`);
    option.value = String(layer); $('similarityLayer').append(option);
  }
  $('segmentCount').textContent = `${experiment.segments.length} SEGMENTS`;
  const segmentation = experiment.segmentation_space ? `Vision encoder · ${experiment.vision_tokens_per_frame} patches/帧 · ${experiment.vision_feature_dim} 维 · 投影/池化前` : '旧结果：Key 分段（重新生成可使用 vision encoder）';
  $('segmentMeta').textContent = `${segmentation} · EMA α=${experiment.alpha} · 阈值倍数 ${experiment.multiplier} · 切分前至少 ${experiment.min_frames} 帧`;
  const {segments, layer_distances, transitions, ...method} = experiment;
  $('segmentMethod').textContent = JSON.stringify({...method, summary_prompts: segments.map(s => ({segment: s.id + 1, prompt: s.summary.prompt}))}, null, 2);
  const table = node('table', 'segment-table'), head = node('tr');
  ['相邻帧', '距离 D', '1 / (1 + D)', '此前 EMA', '阈值', '判定'].forEach(label => head.append(node('th', '', label)));
  table.append(head);
  transitions.forEach(t => {
    const row = node('tr', t.boundary ? 'boundary-row' : '');
    const reason = {warmup: '初始化', boundary: '开启新段', min_frames: '未达最少帧数', below_threshold: '同一段'}[t.reason];
    [`${t.previous_frame + 1} → ${t.current_frame + 1}`, fmt(t.distance, 5), fmt(t.similarity, 5), fmt(t.ema_before, 5), fmt(t.threshold, 5), reason].forEach(value => row.append(node('td', '', value)));
    table.append(row);
  });
  $('segmentTransitions').replaceChildren(table);
  renderSegmentDistance(transitions);
  $('segmentCards').replaceChildren();
  segments.forEach(segment => {
    const card = node('article', 'segment-card'); card.id = `segment-${segment.id}`;
    card.append(node('h3', '', `S${segment.id + 1} · 帧 ${segment.start_frame + 1}–${segment.end_frame} · ${fmt(segment.first_timestamp, 2)}–${fmt(segment.last_timestamp, 2)}s`));
    const strip = node('div', 'segment-strip');
    segment.frame_indices.forEach(index => {
      const frame = state.result.frames[index], button = node('button', 'segment-thumbnail'); button.type = 'button';
      const img = node('img'); img.src = frame.image_url; img.alt = `S${segment.id + 1} 帧 ${index + 1}`; img.loading = 'lazy';
      button.append(img, node('span', '', `#${index + 1} · ${fmt(frame.timestamp, 2)}s`));
      button.addEventListener('click', () => selectFrame(index, true)); strip.append(button);
    });
    const summary = segment.summary;
    card.append(strip, node('p', 'answer-text', summary.text || '（模型没有生成可显示文字；图片 token 的平均相似度未定义。）'));
    const reason = {eos: '完成', max_new_tokens: '达到长度上限', context_limit: '达到上下文上限'}[summary.finish_reason];
    card.append(node('p', 'tiny', `${reason} · ${summary.generated_tokens} token · ${fmt(summary.elapsed_seconds, 2)} 秒 · 仅输入该段图片`));
    $('segmentCards').append(card);
  });
  for (const id of ['similaritySummary', 'similaritySegment']) {
    $(id).replaceChildren();
    segments.forEach(s => {
      const option = node('option', '', `${id === 'similaritySummary' ? 'Summary' : 'Segment'} ${s.id + 1} · ${fmt(s.first_timestamp, 1)}–${fmt(s.last_timestamp, 1)}s`);
      option.value = s.id; $(id).append(option);
    });
  }
  tokenSimilarity.col = 0;
  fillSimilarityFrames();
}
function renderSegmentDistance(transitions) {
  const container = $('segmentDistanceChart'); container.replaceChildren();
  if (!transitions.length) { container.append(node('p', 'muted', '单帧输入，没有相邻帧距离。')); return; }
  const svg = svgNode('svg', {viewBox: '0 0 900 195', role: 'img', 'aria-label': '相邻帧归一化视觉特征距离和历史 EMA 阈值；橙色点表示新段边界'});
  const max = Math.max(...transitions.flatMap(t => [t.distance, t.threshold || 0]), 1e-6) * 1.1;
  const x = i => transitions.length === 1 ? 465 : 65 + i * 790 / (transitions.length - 1);
  const y = value => 150 - value / max * 110;
  svg.append(svgNode('text', {x: 65, y: 18}, '青色：帧距离 D　虚线：此前 EMA（倍数 1）　橙色点：新段起始'));
  for (const f of [0, .5, 1]) {
    svg.append(svgNode('line', {x1: 65, x2: 855, y1: y(max * f), y2: y(max * f), stroke: '#e6ecef'}));
    svg.append(svgNode('text', {x: 55, y: y(max * f) + 4, 'text-anchor': 'end'}, fmt(max * f)));
  }
  for (const [key, color, dash] of [['distance', '#087f80', ''], ['threshold', '#75889b', '5 4']]) {
    let path = '';
    transitions.forEach((t, i) => { if (t[key] != null) path += `${path ? 'L' : 'M'}${x(i)},${y(t[key])} `; });
    svg.append(svgNode('path', {d: path, fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-dasharray': dash}));
  }
  transitions.forEach((t, i) => {
    const dot = svgNode('circle', {cx: x(i), cy: y(t.distance), r: t.boundary ? 5 : 3, fill: t.boundary ? '#d77a3a' : '#087f80'});
    dot.append(svgNode('title', {}, `帧 ${t.current_frame + 1}: D=${fmt(t.distance, 5)} / 阈值=${fmt(t.threshold, 5)} / ${t.reason}`)); svg.append(dot);
    if (i % Math.max(1, Math.ceil(transitions.length / 8)) === 0 || i === transitions.length - 1)
      svg.append(svgNode('text', {x: x(i), y: 177, 'text-anchor': 'middle'}, `帧 ${t.current_frame + 1}`));
  });
  container.append(svg);
}
function fillSimilarityFrames() {
  const experiment = state.result?.segment_experiment;
  if (!experiment) return;
  const segment = experiment.segments[Number($('similaritySegment').value)];
  $('similarityFrame').replaceChildren();
  segment.frame_indices.forEach((index, local) => {
    const frame = state.result.frames[index];
    const option = node('option', '', `段内 ${local + 1}/${segment.frame_indices.length} · 全局帧 ${index + 1} · ${fmt(frame.timestamp, 2)}s`);
    option.value = index; $('similarityFrame').append(option);
  });
}
function similarityColorRange(scores) {
  const valid = scores.filter(Number.isFinite);
  return valid.length ? {min: Math.min(...valid), max: Math.max(...valid)} : {min: null, max: null};
}
function cosineColor(value, range) {
  if (!Number.isFinite(value) || !Number.isFinite(range?.min) || !Number.isFinite(range?.max)) return '#dce2e5';
  const target = [8, 127, 128];
  const strength = range.max === range.min ? .5 : Math.max(0, Math.min(1, (value - range.min) / (range.max - range.min)));
  return `rgb(${target.map(c => Math.round(255 + (c - 255) * strength)).join(',')})`;
}
function renderSimilarityScale(range) {
  $('similarityScaleMin').textContent = `min ${fmt(range.min, 6)}`;
  $('similarityScaleMax').textContent = `max ${fmt(range.max, 6)}`;
  const empty = range.min == null, constant = !empty && range.min === range.max;
  $('similarityScaleRamp').style.background = empty ? '#dce2e5' : constant ? cosineColor(range.min, range) : 'linear-gradient(90deg, white, rgb(8,127,128))';
  $('similarityScaleNote').textContent = empty ? '无有效分数' : constant ? '所有分数相同 · 统一颜色' : '当前帧实际范围 · 灰色：无有效分数';
}
function meanSummaryTokenScores(data) {
  return Array.from({length: data.image_tokens}, (_, col) => {
    if (!data.cosine.length) return null;
    let sum = 0;
    for (const row of data.cosine) {
      if (!Number.isFinite(row[col])) return null;
      sum += row[col];
    }
    return sum / data.cosine.length;
  });
}
function frameMeanPruning(scores) {
  const valid = scores.filter(Number.isFinite);
  if (!valid.length) return {threshold: null, pruned: scores.map(() => false), valid: 0};
  // Subtract an anchor before summing so identical scores are retained exactly.
  const anchor = valid[0];
  const threshold = anchor + valid.reduce((sum, value) => sum + (value - anchor), 0) / valid.length;
  return {threshold, pruned: scores.map(value => Number.isFinite(value) && value < threshold), valid: valid.length};
}
async function renderSegmentSimilarity() {
  tokenSimilarity.controller?.abort();
  const request = ++tokenSimilarity.request;
  tokenSimilarity.data = null;
  $('exportSegmentCsv').disabled = true;
  $('similarityOverlay').replaceChildren();
  $('similarityPrunedPreview').hidden = true;
  $('similarityPrunedMask').replaceChildren();
  $('similarityPruningMeta').textContent = '';
  renderSimilarityScale({min: null, max: null});
  const experiment = state.result?.segment_experiment;
  if (!experiment) return;
  const summary = Number($('similaritySummary').value), frameIndex = Number($('similarityFrame').value);
  const frame = state.result.frames[frameIndex], layer = $('similarityLayer').value;
  const representation = $('similarityRepresentation').value;
  const representationLabel = representation === 'hidden' ? 'Hidden states' : 'Key states';
  $('similarityRepresentationNote').textContent = representation === 'hidden'
    ? 'Hidden states：所选 decoder block 的输出（attention 与 MLP 残差更新后，最终模型归一化前）。'
    : 'Key states：所选层 k_proj 的输出（RoPE 前），拼接原生 KV heads。';
  $('similaritySummaryText').textContent = experiment.segments[summary].summary.text || '（无可显示文字。）';
  $('similarityImage').src = frame.image_url;
  $('similarityImageStage').style.aspectRatio = `${frame.width} / ${frame.height}`;
  $('similarityPrunedImage').src = frame.image_url;
  $('similarityPrunedStage').style.aspectRatio = `${frame.width} / ${frame.height}`;
  $('similarityFrameMeta').textContent = `全局帧 ${frameIndex + 1} · 源帧 ${frame.source_index} · ${fmt(frame.timestamp, 2)}s · ${frame.width} × ${frame.height}`;
  $('segmentSimilarityMeta').textContent = '正在计算图片 token 的平均相似度…';
  $('segmentSimilarityReadout').textContent = '正在读取…';
  if (!experiment.token_similarity?.url) {
    $('segmentSimilarityMeta').textContent = '旧结果未保存逐 token 表示，请重新生成观察数据。'; return;
  }
  const controller = new AbortController(); tokenSimilarity.controller = controller;
  try {
    const query = new URLSearchParams({summary, frame: frameIndex, layer, representation});
    const response = await fetch(`${experiment.token_similarity.url}?${query}`, {signal: controller.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    if (request !== tokenSimilarity.request) return;
    data.mean_scores = meanSummaryTokenScores(data);
    data.color_range = similarityColorRange(data.mean_scores);
    data.pruning = frameMeanPruning(data.mean_scores);
    tokenSimilarity.data = data;
    tokenSimilarity.col = Math.min(tokenSimilarity.col, data.image_tokens - 1);
    const own = Number($('similaritySegment').value) === summary;
    $('segmentSimilarityMeta').textContent = `${representationLabel} · ${layer === 'all' ? '全部层平均' : `第 ${Number(layer) + 1} 层`} · 每个图片 token 对 ${data.text_tokens.length} 个 summary token 取均值 · ${data.image_tokens} 个图片位置 · ${own ? 'summary 来源段' : '跨段对照'} · 当前帧 min–max 色阶`;
    renderSimilarityScale(data.color_range);
    $('exportSegmentCsv').disabled = !data.text_tokens.length;
    drawMeanSimilarity();
    drawPrunedSimilarity();
    inspectMeanSimilarity(tokenSimilarity.col);
  } catch (error) {
    if (error.name === 'AbortError' || request !== tokenSimilarity.request) return;
    $('segmentSimilarityMeta').textContent = `相似度读取失败：${error.message}`;
    $('segmentSimilarityReadout').textContent = '可重新选择帧重试。';
  }
}
function drawPrunedSimilarity() {
  const data = tokenSimilarity.data; if (!data) return;
  const {threshold, pruned, valid} = data.pruning;
  const mask = $('similarityPrunedMask'); mask.replaceChildren();
  $('similarityPrunedPreview').hidden = threshold == null;
  if (threshold == null) {
    $('similarityPruningMeta').textContent = '无有效相似度分数，无法计算帧内均值或展示剪枝预览。';
    return;
  }
  mask.setAttribute('viewBox', '0 0 1000 1000'); mask.setAttribute('preserveAspectRatio', 'none');
  state.result.spatial.cells.forEach((cell, index) => {
    if (!pruned[index]) return;
    const [x1, y1, x2, y2] = cell.box;
    const rect = svgNode('rect', {x: x1 * 1000, y: y1 * 1000, width: (x2 - x1) * 1000, height: (y2 - y1) * 1000,
      fill: '#fff', 'data-token-index': index});
    rect.append(svgNode('title', {}, `图片 token #${index} · ${fmt(data.mean_scores[index], 6)} < 帧内均值 ${fmt(threshold, 6)}`));
    mask.append(rect);
  });
  const count = pruned.filter(Boolean).length, unknown = data.image_tokens - valid;
  $('similarityPruningMeta').textContent = `帧内平均相似度 ${fmt(threshold, 6)} · 剪去 ${count}/${data.image_tokens} 个 token (${pct(count / data.image_tokens)}) · 保留 ${valid - count} 个有效 token。严格低于均值才剪枝，等于均值保留。${unknown ? ` ${unknown} 个无效分数不参与均值，保留原图区域。` : ''} 白色仅表示 token 空间区域的遮罩预览。`;
}
function inspectMeanSimilarity(col) {
  const data = tokenSimilarity.data; if (!data) return;
  const cell = state.result.spatial.cells[col], frame = state.result.frames[data.frame_index];
  const location = cell ? `空间行 ${cell.row + 1} / 列 ${cell.col + 1}` : `帧内 token #${col}`;
  $('segmentSimilarityReadout').textContent = `Summary ${data.summary_index + 1} · 全部 ${data.text_tokens.length} 个内容 token\n帧 ${data.frame_index + 1} (${fmt(frame.timestamp, 2)}s) · 图片 token #${col} · 全局 token #${data.frame_token_start + col} · ${location}\n平均余弦相似度：${fmt(data.mean_scores[col], 6)}${!data.text_tokens.length ? '（summary 无内容 token，均值未定义）' : ''}`;
  $('similarityOverlay').querySelectorAll('rect').forEach((rect, index) => {
    rect.setAttribute('stroke', index === col ? '#ff8a24' : '#ffffff55');
    rect.setAttribute('stroke-width', index === col ? 5 : 1);
  });
}
function drawMeanSimilarity() {
  const data = tokenSimilarity.data; if (!data) return;
  const overlay = $('similarityOverlay'); overlay.replaceChildren();
  overlay.setAttribute('viewBox', '0 0 1000 1000'); overlay.setAttribute('preserveAspectRatio', 'none');
  state.result.spatial.cells.forEach((c, index) => {
    const [x1, y1, x2, y2] = c.box;
    const rect = svgNode('rect', {x: x1 * 1000, y: y1 * 1000, width: (x2 - x1) * 1000, height: (y2 - y1) * 1000,
      fill: cosineColor(data.mean_scores[index], data.color_range), 'fill-opacity': .5, stroke: '#ffffff55', 'stroke-width': 1});
    rect.append(svgNode('title', {}, `图片 token #${index} · 全部 ${data.text_tokens.length} 个 summary token 的平均相似度：${fmt(data.mean_scores[index], 6)}`));
    rect.addEventListener('mouseenter', () => inspectMeanSimilarity(index));
    rect.addEventListener('click', () => { tokenSimilarity.col = index; inspectMeanSimilarity(index); });
    overlay.append(rect);
  });
}
$('similarityOverlay').addEventListener('mouseleave', () => inspectMeanSimilarity(tokenSimilarity.col));
$('similarityOverlay').addEventListener('keydown', event => {
  const data = tokenSimilarity.data; if (!data) return;
  const cols = state.result.spatial.cols || 14;
  const moves = {ArrowLeft: -1, ArrowRight: 1, ArrowUp: -cols, ArrowDown: cols};
  if (!(event.key in moves)) return;
  event.preventDefault();
  tokenSimilarity.col = Math.max(0, Math.min(data.image_tokens - 1, tokenSimilarity.col + moves[event.key]));
  inspectMeanSimilarity(tokenSimilarity.col);
});
$('similaritySummary').addEventListener('change', () => {
  tokenSimilarity.col = 0;
  $('similaritySegment').value = $('similaritySummary').value;
  fillSimilarityFrames(); renderSegmentSimilarity();
});
$('similaritySegment').addEventListener('change', () => { fillSimilarityFrames(); renderSegmentSimilarity(); });
$('similarityFrame').addEventListener('change', renderSegmentSimilarity);
$('similarityLayer').addEventListener('change', renderSegmentSimilarity);
$('similarityRepresentation').addEventListener('change', renderSegmentSimilarity);
$('exportSegmentCsv').addEventListener('click', () => {
  const data = tokenSimilarity.data; if (!data?.text_tokens.length) return;
  const header = 'summary_index,frame_index,layer,representation,summary_token_count,image_token,global_token,grid_row,grid_col,mean_cosine,frame_mean_cosine,pruned';
  const rows = data.mean_scores.map((value, j) => {
    const cell = state.result.spatial.cells[j];
    return [data.summary_index, data.frame_index, data.layer, data.representation, data.text_tokens.length, j,
      data.frame_token_start + j, cell?.row ?? '', cell?.col ?? '', value ?? '',
      data.pruning.threshold ?? '', Number.isFinite(value) ? Number(data.pruning.pruned[j]) : ''].join(',');
  });
  const url = URL.createObjectURL(new Blob([header + '\n' + rows.join('\n')], {type:'text/csv;charset=utf-8'}));
  const link = node('a'); link.href = url; link.download = `summary-${data.summary_index + 1}-frame-${data.frame_index + 1}-${data.representation}-layer-${data.layer}-mean.csv`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
async function init() {
  try {
    [state.config, state.samples] = await Promise.all([api('/api/config'), api('/api/samples')]);
    $('modelInfo').textContent = `${state.config.model_path.split('/').pop()} · ${state.config.device}`;
    $('maxFrames').max = state.config.max_frames;
    $('segmentMinFrames').max = state.config.max_frames;
    $('maxFrames').value = Math.min(8, state.config.max_frames);
    $('videoWindowSize').value = state.config.default_video_window_size ?? 32;
    updateWindowLabels();
    if (!state.samples.length) throw new Error('RVS 数据集中没有样本');
    $('split').value = state.samples[0].split;
    fillSamples(); await loadSample(false);
  } catch (error) { showError(error); $('run').disabled = true; }
}
init();
