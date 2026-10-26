'use strict';
const cachePlayback = {timer: null, live: null, revision: null, liveSide: null, stopped: false, rowSignatures: new Map()};
const mib = value => value == null ? '—' : `${fmt(value / 1048576, 2)} MiB`;
const millis = value => value == null ? '—' : `${fmt(value * 1000, 2)} ms`;
const deltaMs = value => value == null ? '—' : `${value > 0 ? '+' : ''}${fmt(value * 1000, 2)} ms`;
function compressionTable(headers, rows) {
  const table = node('table', 'compression-table'), head = node('thead'), tr = node('tr');
  headers.forEach(text => tr.append(node('th', '', text))); head.append(tr); table.append(head);
  const body = node('tbody');
  rows.forEach(values => { const row = node('tr'); values.forEach(text => row.append(node('td', '', text))); body.append(row); });
  table.append(body); return table;
}
function renderCompressionResult(result, url, cached) {
  stopCachePlayback(); cachePlayback.live = null; cachePlayback.revision = null; cachePlayback.rowSignatures.clear();
  $('compressionResults').classList.remove('is-live');
  $('compressionStateFollowLabel').hidden = true;
  const e = result.compression_experiment, before = e.runs.before, after = e.runs.after;
  $('results').hidden = true; $('emptyState').hidden = true; $('compressionResults').hidden = false;
  const beforeStats = before.model_stats ?? {}, afterStats = after.model_stats ?? {};
  const summarizer = e.config?.summarizer_model_path ?? null;
  $('compressionLabel').textContent = `${e.frame_count} 帧 · ${e.method.dtype ?? 'float16'} · ${beforeStats.model_instances ?? 1} → ${afterStats.model_instances ?? e.method.model_instances ?? 1} 个模型 / 顺序执行 · 摘要配置：${summarizer ?? '主模型'} · 重试 ${afterStats.summary_retries ?? 0} 次 / 最终失败 ${afterStats.summary_failures ?? 0} 次 · ${cached ? '缓存结果' : fmt(result.total_elapsed_seconds, 1) + ' 秒'}`;
  $('compressionJson').hidden = false; $('compressionJson').href = url;
  const memory = e.memory, saved = memory.persistent_kv;
  const gpu = e.method.cache_device.startsWith('cuda');
  $('compressionCards').replaceChildren(
    metricCard(gpu ? '持久 KV 节省显存' : '持久 KV 节省内存', mib(saved?.saved_bytes), saved ? `减少 ${pct(saved.saved_fraction)} · 含量化参数与索引` : '对照不完整，无法计算'),
    metricCard('运行峰值显存差值', mib(memory.cuda_peak?.saved_bytes), '压缩前 − 压缩后；负数表示峰值增加'),
    metricCard('压缩事件 / 淘汰帧', after.status === 'complete' ? `${after.cache.counts.compressions} / ${after.cache.counts.evicted_frames}` : '—', after.status === 'complete' ? `预算 ${mib(after.cache.budget_bytes)} · 保留 ${after.cache.retained_frames} 帧` : after.error));
  const rows = [
    ['持久 KV 张量', memory.persistent_kv],
    ['编码完成时 CUDA 分配量', memory.cuda_allocated],
    ['整个运行的 CUDA 分配峰值', memory.cuda_peak],
  ].map(([label, item]) => [label, mib(item?.before_bytes), mib(item?.after_bytes), mib(item?.saved_bytes), pct(item?.saved_fraction)]);
  if (beforeStats.model_parameter_bytes != null && afterStats.model_parameter_bytes != null) {
    rows.push(['模型参数（含摘要模型）', mib(beforeStats.model_parameter_bytes), mib(afterStats.model_parameter_bytes),
      mib(beforeStats.model_parameter_bytes - afterStats.model_parameter_bytes), '—']);
  }
  $('compressionMemoryTable').replaceChildren(compressionTable(['统计口径', '压缩前', '压缩后', '节省量', '节省比例'], rows));
  $('compressionMemoryNote').textContent = '两侧独立清空缓存后依次运行。持久 KV 不含模型权重；CUDA 分配量和峰值包含权重及工作空间，包括压缩侧独立摘要模型；allocator reserved 不计作节省。';
  prepareCompressionQuestions(e);
  renderCompressionAnswer();
  renderRetrievalHits();
  $('compressionCostTable').replaceChildren(compressionTable(['阶段', '压缩前', '压缩后', '新增耗时'], e.stages.map(stage => [stage.label, millis(stage.before_seconds), millis(stage.after_seconds), deltaMs(stage.delta_seconds)])));
  $('compressionTimingNote').textContent = `不含模型加载及视频解码；摘要模型加载 ${fmt(after.summarizer_load_seconds ?? 0, 3)} 秒（计入任务总时间）。单次顺序测量，先运行原精度，首次调用的 kernel 初始化可能影响时间。编码总耗时：${fmt(before.encoding_seconds, 3)} → ${fmt(after.encoding_seconds, 3)} 秒。未触发的策略阶段显示 0。`;
  $('compressionMethod').textContent = JSON.stringify({method: e.method, config: e.config, model_stats: {before: beforeStats, after: afterStats}, question_time: e.cutoff_timestamp, commits: after.commits ?? []}, null, 2);
  const summaries = (after.commits ?? []).filter(commit => 'summary' in commit);
  $('compressionSummariesPanel').hidden = !summaries.length;
  $('compressionSummariesTable').replaceChildren(compressionTable(
    ['Segment', '实际摘要模型', '摘要', '小模型错误', '主模型重试', '最终错误'],
    summaries.map(commit => [commit.segment_id, commit.summary_source ?? '主模型', commit.summary || '（空摘要）',
      commit.summarizer_error || '—', commit.summary_retried ? (commit.summary_retry_succeeded ? '成功' : '失败') : '未重试',
      commit.summary_error || commit.score_error || '—'])));
  $('compressionFrames').replaceChildren();
  result.frames.forEach(frame => {
    const figure = node('figure', 'compression-frame'), img = node('img'); img.src = frame.image_url; img.alt = `采样帧 ${frame.index+1}`; img.loading = 'lazy';
    figure.append(img, node('figcaption', 'muted', `帧 ${frame.index+1} · ${fmt(frame.timestamp, 2)}s`)); $('compressionFrames').append(figure);
  });
  prepareCompressionFrameStates(e);
  renderCompressionCostChart(e.stages);
  renderCompressionMemoryChart(e);
  renderCompressionFrameStates();
}
function csvg(tag, attrs = {}, text) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
  if (text != null) el.textContent = text;
  return el;
}
function compressionQuestions(e) {
  return e.questions ?? [{question_id: 1, question_index: e.question_index, question: e.question}];
}
function compressionAnswers(run) { return run.answers ?? (run.answer ? [run.answer] : []); }
function compressionQuestionTime(q) {
  return q.question_index == null ? '自定义问题，无数据集时间' :
    datasetQuestionTime(state.compression?.sample?.conversations?.[q.question_index]);
}
function prepareCompressionQuestions(e) {
  $('compressionAnswerQuestion').replaceChildren();
  const all = node('option', '', '全部问题 · 累计命中'); all.value = 'all';
  $('compressionHitQuestion').replaceChildren(all);
  compressionQuestions(e).forEach((q, i) => {
    for (const id of ['compressionAnswerQuestion', 'compressionHitQuestion']) {
      const option = node('option', '', `Q${i+1} · ${compressionQuestionTime(q)} · ${q.question}`); option.value = i; $(id).append(option);
    }
  });
}
function renderCompressionAnswer() {
  const result = state.compression; if (!result) return;
  const e = result.compression_experiment, index = Number($('compressionAnswerQuestion').value);
  const q = compressionQuestions(e)[index]; if (!q) return;
  $('compressionQuestion').textContent = q.question;
  $('compressionContext').textContent = `${compressionQuestionTime(q)}。共 ${compressionQuestions(e).length} 题，两侧各编码同一批 ${e.frame_count} 帧一次，在 ${fmt(e.cutoff_timestamp, 2)} 秒处依次独立提问。`;
  for (const [side, run] of [['Before', e.runs.before], ['After', e.runs.after]]) {
    const answer = compressionAnswers(run)[index];
    $('compression'+side+'Answer').textContent = run.status !== 'complete' ? '本侧失败：'+run.error : answer?.answer || '（未生成可显示文本）';
    $('compression'+side+'Meta').textContent = answer ? `${answer.finish_reason} · ${fmt(answer.qa_seconds, 3)} 秒 · 命中帧 ID：${answer.hit_frames.join(', ') || '无'}` : '';
  }
  $('compressionAnswerStatus').textContent = e.status !== 'complete' ? '对照不完整' :
    compressionAnswers(e.runs.before)[index]?.answer === compressionAnswers(e.runs.after)[index]?.answer ? '回答文本相同' : '回答文本不同';
  $('compressionReference').textContent = q.question_index == null ? '自定义问题，无参考答案' : result.sample.conversations[q.question_index]?.answer ?? '未提供';
}
function retrievalHitSeries(e, kind) {
  const scope = $('compressionHitQuestion').value, key = kind === 'frame' ? 'frame_hits' : 'segment_hits';
  const idKey = kind === 'frame' ? 'frame_id' : 'segment_id';
  return [['before', '原精度', '#8a98a2'], ['after', '压缩缓存', '#087f80']].map(([side, label, color]) => {
    const run = e.runs[side], answer = scope === 'all' ? null : compressionAnswers(run)[Number(scope)];
    const selected = new Set(answer?.[kind === 'frame' ? 'hit_frames' : 'hit_segments'] ?? []);
    const points = (run.retrieval_stats?.[key] ?? []).map(row => ({...row, id: row[idKey],
      hits: scope === 'all' ? row.hits : Number(selected.has(row[idKey]))}));
    return {side, label, color, points};
  });
}
function renderHitChart(target, series, label) {
  target.replaceChildren();
  const ids = [...new Set(series.flatMap(s => s.points.map(p => p.id)))].sort((a,b) => a-b);
  if (!ids.length) { target.append(node('p', 'muted', '暂无可用统计；压缩侧失败时无法确定 Segment 分段。')); return; }
  const maximum = Math.max(1, ...series.flatMap(s => s.points.map(p => p.hits)));
  const svg = csvg('svg', {viewBox: '0 0 780 250', role: 'img', 'aria-label': `${label}检索命中次数折线图`});
  const x = id => ids.length === 1 ? 410 : 65+ids.indexOf(id)*690/(ids.length-1), y = value => 205-value/maximum*170;
  const step = Math.max(1, Math.ceil(maximum/4));
  for (const value of [...new Set([0, ...Array.from({length: Math.floor(maximum/step)}, (_,i)=>(i+1)*step), maximum])]) {
    svg.append(csvg('line', {x1:65, x2:755, y1:y(value), y2:y(value), stroke:'#e2e8eb'}),
      csvg('text', {x:55, y:y(value)+4, 'text-anchor':'end'}, value));
  }
  svg.append(csvg('text', {x:10,y:18}, '命中次数'));
  ids.forEach((id, i) => {
    if (i % Math.max(1, Math.ceil(ids.length/12)) === 0 || i === ids.length-1)
      svg.append(csvg('text', {x:x(id),y:225,'text-anchor':'middle'}, id));
  });
  svg.append(csvg('text', {x:410,y:246,'text-anchor':'middle'}, `${label} ID`));
  for (const s of series) {
    svg.append(csvg('polyline', {points:s.points.map(p=>`${x(p.id)},${y(p.hits)}`).join(' '), fill:'none', stroke:s.color,
      'stroke-width':s.side === 'before' ? 4 : 2, 'stroke-dasharray':s.side === 'before' ? '6 4' : 'none'}));
    for (const p of s.points) {
      const detail = `${s.label} · ${label} ${p.id} · ${p.hits} 次命中` +
        (p.timestamp != null ? ` · ${fmt(p.timestamp, 2)}s · Segment ${p.segment_id ?? '未知'}` : ` · 帧 ID：${p.frame_ids.join(', ')}`);
      const dot = csvg('circle', {cx:x(p.id), cy:y(p.hits), r:s.side === 'before' ? 5 : 3, fill:s.color, tabindex:0, 'aria-label':detail});
      dot.append(csvg('title', {}, detail));
      for (const event of ['mouseenter', 'focus', 'click']) dot.addEventListener(event, () => $('compressionHitsReadout').textContent = detail);
      svg.append(dot);
    }
  }
  target.append(svg, node('p', 'tiny', series.map(s => `${s.side === 'before' ? '灰色虚线' : '青色实线'}：${s.label}${s.points.length ? '' : '（本侧统计不可用）'}`).join(' · ')));
}
function retrievalHitRows(e) {
  return ['frame', 'segment'].flatMap(kind => retrievalHitSeries(e, kind).flatMap(s => s.points.map(p =>
    [kind, p.id, s.side, p.hits, p.timestamp ?? '', p.segment_id ?? '', p.frame_ids?.join(' ') ?? ''])));
}
function renderRetrievalHits() {
  const e = state.compression?.compression_experiment; if (!e) return;
  $('compressionHitsNote').textContent = `共 ${compressionQuestions(e).length} 题；每题每帧最多命中一次（跨层去重，含 recent 帧）。Segment 中任意帧命中即计一次。两侧统一按压缩侧的分段统计；已淘汰帧仍保留在横轴。`;
  $('compressionHitsReadout').textContent = '悬停或聚焦曲线上的点查看命中次数。';
  renderHitChart($('compressionFrameHitsChart'), retrievalHitSeries(e, 'frame'), '帧');
  renderHitChart($('compressionSegmentHitsChart'), retrievalHitSeries(e, 'segment'), 'Segment');
  $('compressionHitsTable').replaceChildren(compressionTable(['类型', 'ID', '缓存版本', '命中次数', '时间 / 秒', 'Segment ID', '段内帧 ID'], retrievalHitRows(e)));
}
$('compressionAnswerQuestion').addEventListener('change', renderCompressionAnswer);
$('compressionHitQuestion').addEventListener('change', renderRetrievalHits);
$('compressionHitsCsv').addEventListener('click', () => {
  const e = state.compression?.compression_experiment; if (!e) return;
  const csv = ['kind,id,side,hits,timestamp,segment_id,frame_ids', ...retrievalHitRows(e).map(row => row.join(','))].join('\n');
  const url = URL.createObjectURL(new Blob(['\ufeff'+csv], {type:'text/csv;charset=utf-8'}));
  const link = node('a'); link.href=url; link.download=`retrieval-hits-${$('compressionHitQuestion').value}.csv`; link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
});
function renderCompressionMemoryChart(e) {
  const target = $('compressionMemoryChart'); target.replaceChildren();
  const left = e.runs.before.timeline, right = e.runs.after.timeline;
  if (!left || !right) { $('compressionFrame').disabled = true; $('compressionFrameReadout').textContent = '对照不完整，无法绘制显存轨迹。'; return; }
  const max = Math.max(1, ...left.map(p => p.persistent_tensor_bytes), ...right.map(p => p.persistent_tensor_bytes), e.runs.after.cache.budget_bytes)*1.12;
  const svg = csvg('svg', {viewBox: '0 0 780 250', role: 'img', 'aria-label': '随输入帧变化的原精度与压缩 KV 张量内存对比'});
  const x = index => 65 + index*690/Math.max(1, left.length-1), y = bytes => 210-bytes/max*180;
  for (let i=0; i<=4; i++) {
    const value = max*i/4;
    svg.append(csvg('line', {x1:65, x2:755, y1:y(value), y2:y(value), stroke:'#e2e8eb'}));
    svg.append(csvg('text', {x:58, y:y(value)+4, 'text-anchor':'end'}, fmt(value/1048576, 1)));
  }
  svg.append(csvg('text', {x:10, y:18}, 'MiB'));
  const budgetY = y(e.runs.after.cache.budget_bytes);
  svg.append(csvg('line', {x1:65, x2:755, y1:budgetY, y2:budgetY, stroke:'#d77a3a', 'stroke-dasharray':'5 4'}));
  svg.append(csvg('text', {x:750, y:budgetY-7, 'text-anchor':'end'}, '预算（含元数据）'));
  [[left, '#8a98a2'], [right, '#087f80']].forEach(([points, color]) => {
    svg.append(csvg('polyline', {points:points.map((p,i)=>`${x(i)},${y(p.persistent_tensor_bytes)}`).join(' '), fill:'none', stroke:color, 'stroke-width':2.5}));
    points.forEach((p,i) => {
      const dot=csvg('circle', {cx:x(i), cy:y(p.persistent_tensor_bytes), r:3, fill:color, 'data-point':i});
      dot.append(csvg('title', {}, `帧 ${p.frame_index+1} / ${p.kind==='flush'?'末段提交':'帧入库'}：${mib(p.persistent_tensor_bytes)}`));
      dot.addEventListener('click', () => setCacheProgress(i, true)); svg.append(dot);
    });
  });
  svg.append(csvg('text', {x:65, y:238}, '第一帧'), csvg('text', {x:755, y:238, 'text-anchor':'end'}, '末段提交后'));
  target.append(svg, node('p', 'tiny', '灰色：原精度 · 青色：压缩缓存 · 点击曲线或移动下方滑块查看精确值'));
  $('compressionFrame').disabled=false; $('compressionFrame').max=left.length-1; $('compressionFrame').value=left.length-1;
  updateCompressionReadout();
}
function updateCompressionReadout() {
  renderCompressionFrameStates();
  const e=state.compression?.compression_experiment; if (!e?.runs.before.timeline || !e?.runs.after.timeline) return;
  const index=Number($('compressionFrame').value), before=e.runs.before.timeline[index], after=e.runs.after.timeline[index];
  $('compressionFrameReadout').textContent=`帧 ${before.frame_index+1} · ${fmt(before.timestamp,2)}s · ${before.kind==='flush'?'末段提交完成':'当前帧入库后'}\n原精度 ${mib(before.persistent_tensor_bytes)} → 压缩 ${mib(after.persistent_tensor_bytes)} · 节省 ${mib(before.persistent_tensor_bytes-after.persistent_tensor_bytes)}\n保留帧数 ${before.retained_frames} → ${after.retained_frames}`;
  $('compressionMemoryChart').querySelectorAll('[data-point]').forEach(dot=>dot.setAttribute('r', Number(dot.dataset.point)===index?5:3));
}
function renderCompressionCostChart(stages) {
  const target=$('compressionCostChart'); target.replaceChildren();
  const selected=stages.filter(stage=>stage.algorithm), max=Math.max(.001,...selected.map(s=>s.after_seconds??0));
  selected.forEach(stage=>{
    const row=node('div','compression-cost-row'), track=node('div','compression-cost-track'), bar=node('div','compression-cost-bar');
    bar.style.width=`${Math.max(0,stage.after_seconds??0)/max*100}%`; track.append(bar);
    row.append(node('span','',stage.label),track,node('span','',millis(stage.after_seconds))); target.append(row);
  });
}
$('compressionFrame').addEventListener('input', () => { stopCachePlayback(); pauseLiveFollowing(); updateCompressionReadout(); });
$('compressionCsv').addEventListener('click', () => {
  const stages=state.compression?.compression_experiment.stages; if (!stages) return;
  const csv=['stage,before_seconds,after_seconds,delta_seconds', ...stages.map(s=>[s.label,s.before_seconds??'',s.after_seconds??'',s.delta_seconds??''].join(','))].join('\n');
  const url=URL.createObjectURL(new Blob(['\ufeff'+csv],{type:'text/csv;charset=utf-8'}));
  const link=node('a');link.href=url;link.download='kv-compression-costs.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});


function prepareCompressionFrameStates(experiment) {
  const snapshots = experiment.runs.after.timeline ?? experiment.runs.before.timeline ?? [];
  const sample = snapshots.find(point => point.frame_states)?.frame_states;
  $('compressionFrame').max = Math.max(0, snapshots.length-1);
  $('compressionFrame').value = Math.max(0, snapshots.length-1);
  const select = $('compressionStateLayer');
  select.replaceChildren(node('option', '', '全部层 · 每层平均')); select.options[0].value = 'all';
  for (let i = 0; i < (sample?.layer_count ?? 0); i++) {
    const option = node('option', '', `第 ${i+1} 层`); option.value = i; select.append(option);
  }
  select.value = 'all'; select.disabled = !sample;
  $('compressionStateSide').value = experiment.runs.after.status === 'complete' ? 'after' : 'before';
  $('compressionStateFinal').disabled = !sample;
}
function selectedFrameState() {
  const result = state.compression ?? cachePlayback.live;
  if (!result) return null;
  const side = $('compressionStateSide').value;
  const timeline = result.compression_experiment.runs[side]?.timeline;
  if (!timeline?.length) return null;
  const index = Math.min(Number($('compressionFrame').value), timeline.length-1);
  const point = timeline[index];
  if (!point.frame_states) return null;
  return {result, side, index, point, snapshot: point.frame_states, layer: $('compressionStateLayer').value};
}
function frameStateNumbers(snapshot, record, layer) {
  if (!record) return null; // Not encoded yet; distinguish this from an evicted frame.
  const all = layer === 'all';
  const rows = record.counts ?? Array.from({length: snapshot.layer_count}, () => [0, 0, 0, 0]);
  const selected = all ? rows : [rows[Number(layer)]];
  const counts = [0, 0, 0, 0];
  selected.forEach(row => row.forEach((count, i) => { counts[i] += count; }));
  const remaining = counts.reduce((sum, n) => sum+n, 0);
  const bitSum = counts.reduce((sum, n, i) => sum+n*[snapshot.full_precision_dtype === 'float32' ? 32 : 16, 8, 4, 2][i], 0);
  const totals = selected.map(row => row.reduce((sum, n) => sum+n, 0));
  return {counts: counts.map(n => n/selected.length), remaining: remaining/selected.length,
    original: snapshot.tokens_per_frame, deleted: snapshot.tokens_per_frame-remaining/selected.length,
    min: Math.min(...totals), max: Math.max(...totals), bits: remaining ? bitSum/remaining : null};
}
function cacheStateLabel(record, numbers) {
  if (!record) return '未编码';
  if (record.status === 'evicted') return '已整帧淘汰';
  if (record.protected) return record.status === 'open' ? 'Recent 保护 · 未闭段' : 'Recent 保护';
  if (record.status === 'open') return '未闭段 · 原精度';
  if (numbers.counts.slice(1).some(n => n > 0)) return '混合精度';
  if (numbers.deleted > 0) return '已删低分 token';
  return '原精度保留';
}
const tokenCount = n => n == null ? '—' : Number.isInteger(n) ? String(n) : fmt(n, 1);
function renderCompressionFrameStates() {
  const selected = selectedFrameState(), target = $('compressionFrameStates');
  target.replaceChildren();
  $('compressionFrameStatesCsv').disabled = !selected;
  updateCachePlaybackControls(selected);
  if (!selected) {
    $('compressionFrameStatesMeta').textContent = '本侧没有逐帧缓存状态。旧缓存需重新生成压缩对照。';
    return;
  }
  const {result, side, point, snapshot, layer} = selected;
  const records = new Map(snapshot.frames.map(frame => [frame.frame_id, frame]));
  $('compressionFrameStatesMeta').textContent = `${side==='after'?'压缩后':'压缩前'} · 输入到帧 ${point.frame_index+1} (${fmt(point.timestamp,2)}s) · ${point.kind==='flush'?'末段提交完成':'当前帧入库后'} · ${layer==='all'?`全部 ${snapshot.layer_count} 层，每层平均`:`第 ${Number(layer)+1} 层`}`;
  const table = compressionTable(['帧 / 时间', '状态', '剩余 / 原始 token', '精度分布', '已删除 token', '剩余平均位宽'], []);
  table.classList.add('cache-state-table');
  let previousSegment = null;
  result.frames.forEach((frame, index) => {
    const record = records.get(index), numbers = frameStateNumbers(snapshot, record, layer);
    const row = node('tr'); row.dataset.frameId = index; row.dataset.cacheStatus = record?.status ?? 'pending';
    row.dataset.remainingTokens = numbers?.remaining ?? ''; row.dataset.bitCounts = JSON.stringify(numbers?.counts ?? null);
    const segmentId = record?.segment_id;
    if (Number.isInteger(segmentId)) {
      const hue = (190+segmentId*137.508)%360;
      row.dataset.segmentId = segmentId;
      row.style.setProperty('--segment-bg', `hsl(${hue} 55% 96%)`);
      row.style.setProperty('--segment-accent', `hsl(${hue} 44% 62%)`);
      if (previousSegment !== segmentId) row.classList.add('cache-segment-start');
    }
    previousSegment = segmentId;
    const signatureKey = `${side}/${layer}/${index}`, signature = JSON.stringify([record?.status,record?.protected,record?.counts]);
    if (cachePlayback.rowSignatures.has(signatureKey) && cachePlayback.rowSignatures.get(signatureKey)!==signature) row.classList.add('cache-state-changed');
    cachePlayback.rowSignatures.set(signatureKey,signature);

    const info = node('td'), preview = node('img', 'cache-state-thumb');
    preview.src = frame.image_url; preview.alt = `帧 ${index+1}`; preview.loading = 'lazy';
    info.append(preview, node('span', '', `帧 ${index+1} · ${fmt(frame.timestamp,2)}s`));
    if (Number.isInteger(segmentId)) info.append(node('small', 'cache-segment-label', `Segment ${segmentId+1}`));
    else if (record && side==='before') info.append(node('small', 'cache-state-range', '基线未分段'));
    const statusCell = node('td'); statusCell.append(node('span', record?.protected ? 'tag cache-protected' : 'tag', cacheStateLabel(record, numbers)));
    const retained = node('td', '', numbers ? `${tokenCount(numbers.remaining)} / ${numbers.original}` : '—');
    if (numbers && layer==='all') retained.append(node('small', 'cache-state-range', `各层范围 ${numbers.min}–${numbers.max}`));
    const precision = node('td'), bar = node('div', 'cache-state-bar');
    if (numbers) {
      const values = [...numbers.counts, numbers.deleted], labels = ['原精度', '8bit', '4bit', '2bit', '已删除'], classes = ['16', '8', '4', '2', 'deleted'];
      values.forEach((count, i) => {
        if (!count) return;
        const segment = node('span', 'cache-bit-'+classes[i]); segment.style.width = `${100*count/numbers.original}%`;
        segment.title = `${labels[i]}：${tokenCount(count)}${layer==='all'?' / 层平均':''}`; bar.append(segment);
      });
      bar.setAttribute('role', 'img'); bar.setAttribute('aria-label', labels.map((label, i)=>`${label} ${tokenCount(values[i])}`).join('，'));
      precision.append(bar, node('small', 'cache-state-counts', numbers.counts.map((n,i)=>`${labels[i]} ${tokenCount(n)}`).join(' · ')));
    } else { bar.classList.add('cache-state-pending'); precision.append(bar, node('small', 'muted', '尚未进入缓存')); }
    row.append(info, statusCell, retained, precision, node('td', '', tokenCount(numbers?.deleted)), node('td', '', numbers?.bits == null ? '—' : `${fmt(numbers.bits, 2)} bit`));
    table.tBodies[0].append(row);
  });
  target.append(table);
}
$('compressionStateLayer').addEventListener('change', renderCompressionFrameStates);
$('compressionStateSide').addEventListener('change', () => {
  stopCachePlayback(); pauseLiveFollowing(); const selected=selectedFrameState();
  if(selected) { $('compressionFrame').max=selected.result.compression_experiment.runs[selected.side].timeline.length-1; $('compressionFrame').value=selected.index; }
  updateCompressionReadout();
});
$('compressionStateFinal').addEventListener('click', () => {
  stopCachePlayback();
  if(cachePlayback.live && !cachePlayback.stopped) { $('compressionStateFollow').checked=true; followLatestCacheState(); return; }
  const selected = selectedFrameState(); if (!selected) return;
  $('compressionFrame').value = selected.result.compression_experiment.runs[selected.side].timeline.length-1;
  updateCompressionReadout();
});
$('compressionFrameStatesCsv').addEventListener('click', () => {
  const selected = selectedFrameState(); if (!selected) return;
  const {result, side, point, snapshot, layer} = selected, records = new Map(snapshot.frames.map(frame=>[frame.frame_id,frame]));
  const rows = ['side,progress_frame,progress_kind,frame_id,timestamp,segment_id,layer,status,protected,remaining_tokens,original_tokens,fp16_tokens,int8_tokens,int4_tokens,int2_tokens,deleted_tokens,mean_bits'];
  result.frames.forEach((frame, index) => {
    const record = records.get(index), numbers = frameStateNumbers(snapshot, record, layer);
    rows.push([side,point.frame_index,point.kind,index,frame.timestamp,record?.segment_id??'',layer==='all'?'mean':layer,record?.status??'pending',Number(record?.protected??false),numbers?.remaining??'',numbers?.original??'',...(numbers?.counts??['','','','']),numbers?.deleted??'',numbers?.bits??''].join(','));
  });
  const url = URL.createObjectURL(new Blob(['\ufeff'+rows.join('\n')], {type:'text/csv;charset=utf-8'}));
  const link = node('a'); link.href=url; link.download='kv-frame-states.csv'; link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
});


function stopCachePlayback() {
  if(cachePlayback.timer!==null) clearInterval(cachePlayback.timer);
  cachePlayback.timer=null;
  $('compressionStatePlay').textContent='播放'; $('compressionStatePlay').setAttribute('aria-pressed','false');
}
function pauseLiveFollowing() { if(cachePlayback.live) $('compressionStateFollow').checked=false; }
function setCacheProgress(index, manual=false) {
  if(manual) { stopCachePlayback(); pauseLiveFollowing(); }
  const selected=selectedFrameState(); if(!selected) return;
  const last=selected.result.compression_experiment.runs[selected.side].timeline.length-1;
  $('compressionFrame').max=last; $('compressionFrame').value=Math.max(0,Math.min(last,index));
  updateCompressionReadout();
}
function updateCachePlaybackControls(selected) {
  const last=selected ? selected.result.compression_experiment.runs[selected.side].timeline.length-1 : 0;
  $('compressionStateProgress').max=last; $('compressionStateProgress').value=selected?.index??0;
  $('compressionStateProgress').disabled=!selected;
  $('compressionStatePrev').disabled=!selected || selected.index===0;
  $('compressionStateNext').disabled=!selected || selected.index===last;
  $('compressionStatePlay').disabled=!state.compression || !selected || last<1;
  const prefix=cachePlayback.live ? (cachePlayback.stopped?'任务已停止 · 已收到的状态':$('compressionStateFollow').checked?'实时跟随 · 每秒刷新最新完成帧':'已暂停跟随 · 可查看已收到的状态') : cachePlayback.timer!==null?'正在回放':'可播放或逐步查看';
  $('compressionStatePlaybackStatus').textContent=`${prefix}${selected?` · ${selected.index+1}/${last+1} 个状态点`:''}。颜色变化对应实际快照，不插值推算 token 数。`;
}
function startCachePlayback() {
  const selected=selectedFrameState(); if(!state.compression || !selected) return;
  const last=selected.result.compression_experiment.runs[selected.side].timeline.length-1;
  if(last<1) return;
  if(selected.index>=last) setCacheProgress(0);
  cachePlayback.timer=setInterval(()=>{
    const current=selectedFrameState();
    if(!current) { stopCachePlayback(); return; }
    const end=current.result.compression_experiment.runs[current.side].timeline.length-1;
    setCacheProgress(current.index+1);
    if(current.index+1>=end) { stopCachePlayback(); renderCompressionFrameStates(); }
  },Number($('compressionStateSpeed').value));
  $('compressionStatePlay').textContent='暂停'; $('compressionStatePlay').setAttribute('aria-pressed','true');
  updateCachePlaybackControls(selectedFrameState());
}
$('compressionStatePlay').addEventListener('click',()=>{
  if(cachePlayback.timer!==null) { stopCachePlayback(); renderCompressionFrameStates(); } else startCachePlayback();
});
$('compressionStateSpeed').addEventListener('change',()=>{if(cachePlayback.timer!==null){stopCachePlayback();startCachePlayback();}});
$('compressionStatePrev').addEventListener('click',()=>setCacheProgress(Number($('compressionFrame').value)-1,true));
$('compressionStateNext').addEventListener('click',()=>setCacheProgress(Number($('compressionFrame').value)+1,true));
$('compressionStateProgress').addEventListener('input',e=>setCacheProgress(Number(e.target.value),true));
function followLatestCacheState() {
  const live=cachePlayback.live, side=cachePlayback.liveSide; if(!live||!side) return;
  $('compressionStateSide').value=side;
  const timeline=live.compression_experiment.runs[side].timeline;
  $('compressionFrame').max=timeline.length-1; $('compressionFrame').value=timeline.length-1;
  updateCompressionReadout();
}
$('compressionStateFollow').addEventListener('change',()=>{
  if($('compressionStateFollow').checked) followLatestCacheState(); else renderCompressionFrameStates();
});
function renderCompressionLive(job) {
  const update=job.compression_live;
  if(!update || !job.preview || update.revision===cachePlayback.revision) return;
  if(!cachePlayback.live || cachePlayback.live.job_id!==job.id) {
    stopCachePlayback(); cachePlayback.rowSignatures.clear(); cachePlayback.stopped=false;
    cachePlayback.live={job_id:job.id,frames:job.preview.frames,compression_experiment:{runs:{before:{timeline:[]},after:{timeline:[]}}}};
    $('compressionStateFollow').checked=true; $('compressionStateFollow').disabled=false;
    const select=$('compressionStateLayer'); select.replaceChildren(node('option','','全部层 · 每层平均')); select.options[0].value='all';
    for(let i=0;i<update.point.frame_states.layer_count;i++){const option=node('option','',`第 ${i+1} 层`);option.value=i;select.append(option);}
    select.disabled=false; $('compressionStateFinal').disabled=false;
  }
  cachePlayback.revision=update.revision; cachePlayback.liveSide=update.side;
  const timeline=cachePlayback.live.compression_experiment.runs[update.side].timeline;
  const last=timeline.at(-1);
  if(last && last.frame_index===update.point.frame_index && last.kind===update.point.kind) timeline[timeline.length-1]=update.point;
  else timeline.push(update.point);
  $('results').hidden=true; $('emptyState').hidden=true; $('compressionResults').hidden=false;
  $('compressionResults').classList.add('is-live'); $('compressionJson').hidden=true;
  $('compressionStateFollowLabel').hidden=false;
  $('compressionLabel').textContent=`实时计算 · ${update.side==='after'?'压缩后':'原精度 基线'} · 完成帧 ${update.point.frame_index+1}/${job.preview.frames.length}`;
  if($('compressionStateFollow').checked) followLatestCacheState(); else renderCompressionFrameStates();
}
function finishCompressionLive(status) {
  stopCachePlayback(); if(!cachePlayback.live) return;
  cachePlayback.stopped=true; $('compressionStateFollow').disabled=true;
  $('compressionLabel').textContent=`${status==='cancelled'?'任务已取消':'任务失败'} · 以下为停止前收到的状态，非完整对照结果`;
  renderCompressionFrameStates();
}
document.addEventListener('observation:clear',()=>{
  stopCachePlayback(); cachePlayback.live=null; cachePlayback.revision=null; cachePlayback.liveSide=null; cachePlayback.rowSignatures.clear(); cachePlayback.stopped=false;
  $('compressionResults').classList.remove('is-live'); $('compressionStateFollowLabel').hidden=true;
});
