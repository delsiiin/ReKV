'use strict';
const frameQueryState = {controller: null, request: 0, data: null, selected: 0};
function clearFrameQuery() {
  frameQueryState.controller?.abort(); frameQueryState.request++;
  frameQueryState.data = null; frameQueryState.selected = 0;
  $('frameQueryPanel').hidden = true; $('exportFrameQueryCsv').disabled = true;
  $('frameQueryMaps').replaceChildren();
  $('frameQueryMeta').textContent = ''; $('frameQueryMass').textContent = '';
  $('frameQueryReadout').textContent = '';
}
function frameQueryFrames() {
  const segment = state.result.segment_experiment.segments[Number($('frameQuerySegment').value)];
  $('frameQueryFrame').replaceChildren();
  segment.frame_indices.forEach((index, local) => {
    const frame = state.result.frames[index];
    const option = node('option', '', `段内 ${local+1}/${segment.frame_indices.length} · 帧 ${index+1} · ${fmt(frame.timestamp, 2)}s`);
    option.value = index; $('frameQueryFrame').append(option);
  });
}
function initFrameQuery() {
  clearFrameQuery();
  const experiment = state.result?.segment_experiment;
  if (!experiment) return;
  $('frameQueryPanel').hidden = false;
  const available = Boolean(experiment.frame_query_attention?.url);
  for (const id of ['frameQueryLayer', 'frameQuerySegment', 'frameQueryFrame']) {
    $(id).replaceChildren(); $(id).disabled = !available;
  }
  if (!available) {
    $('frameQueryMeta').textContent = '旧缓存没有帧均值 Query，请重新生成观察数据。'; return;
  }
  const all = node('option', '', '全部层平均'); all.value = 'all'; $('frameQueryLayer').append(all);
  for (let i = 0; i < state.result.model.layers; i++) {
    const option = node('option', '', `第 ${i+1} 层`); option.value = i; $('frameQueryLayer').append(option);
  }
  experiment.segments.forEach(segment => {
    const option = node('option', '', `Segment ${segment.id+1} · ${fmt(segment.first_timestamp, 1)}–${fmt(segment.last_timestamp, 1)}s`);
    option.value = segment.id; $('frameQuerySegment').append(option);
  });
  frameQueryFrames(); renderFrameQuery();
}
function syncAttentionSelection(source) {
  if (!state.result?.segment_experiment?.frame_query_attention?.url ||
      !state.result?.segment_experiment?.segment_attention?.url) return;
  const from = source === 'frame' ? 'frameQuery' : 'segmentAttention';
  const to = source === 'frame' ? 'segmentAttention' : 'frameQuery';
  if ($(to+'Segment').value !== $(from+'Segment').value) {
    $(to+'Segment').value = $(from+'Segment').value;
    source === 'frame' ? attentionFrames() : frameQueryFrames();
  }
  $(to+'Frame').value = $(from+'Frame').value;
  $(to+'Layer').value = $(from+'Layer').value;
  source === 'frame' ? renderSegmentAttention() : renderFrameQuery();
}
async function renderFrameQuery() {
  frameQueryState.controller?.abort(); const request = ++frameQueryState.request;
  frameQueryState.data = null; $('exportFrameQueryCsv').disabled = true;
  $('frameQueryMaps').replaceChildren(); $('frameQueryMass').textContent = '';
  const url = state.result?.segment_experiment?.frame_query_attention?.url;
  if (!url) return;
  const segment = $('frameQuerySegment').value, frame = $('frameQueryFrame').value, layer = $('frameQueryLayer').value;
  $('frameQueryMeta').textContent = '正在计算帧均值 Query 对整个 segment 的分数…';
  $('frameQueryReadout').textContent = '正在读取…';
  const controller = new AbortController(); frameQueryState.controller = controller;
  try {
    const response = await fetch(`${url}?${new URLSearchParams({segment, frame, layer})}`, {signal: controller.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    if (request !== frameQueryState.request) return;
    frameQueryState.data = data;
    $('frameQueryMeta').textContent = `RoPE 前 · ${layer === 'all' ? '全部层平均' : `第 ${Number(layer)+1} 层`} · ${data.query_count} 个帧 Query · ${data.segment_tokens} 个 segment Key`;
    $('frameQueryMass').textContent = `当前帧权重和 ${fmt(data.frame_masses.find(f => f.frame_index === Number(frame))?.mass, 6)} · segment 权重和 ${fmt(data.segment_mass, 6)}`;
    for (const measure of attentionMeasures) renderAttentionMap(data, measure, data,
      {state: frameQueryState, readToken: readFrameQueryToken, target: 'frameQueryMaps'});
    $('exportFrameQueryCsv').disabled = false; readFrameQueryToken(frameQueryState.selected);
  } catch (error) {
    if (error.name === 'AbortError' || request !== frameQueryState.request) return;
    $('frameQueryMeta').textContent = `帧均值 Query 读取失败：${error.message}`;
    $('frameQueryReadout').textContent = '可重新选择帧或层重试。';
  }
}
function readFrameQueryToken(index) {
  const data = frameQueryState.data; if (!data) return;
  const cell = state.result.spatial.cells[index];
  $('frameQueryReadout').textContent = `Segment ${data.segment_index+1} · 帧 ${data.frame_index+1} · 图片 token #${index} · 全局 token #${data.frame_token_start+index}${cell ? ` · 空间行 ${cell.row+1} / 列 ${cell.col+1}` : ''}\nSoftmax 前 ${fmt(data.pre_softmax[index], 8)} · Softmax 后 ${fmt(data.post_softmax[index], 8)}`;
  $('frameQueryMaps').querySelectorAll('.attention-overlay rect').forEach(rect => {
    const selected = Number(rect.dataset.tokenIndex) === index;
    rect.setAttribute('stroke', selected ? '#ff8a24' : '#ffffff55');
    rect.setAttribute('stroke-width', selected ? 5 : 1);
  });
}
function frameQueryCsv(data) {
  const pre = frameMeanPruning(data.pre_softmax), post = frameMeanPruning(data.post_softmax);
  const header = 'segment_index,frame_index,layer,image_token,global_token,grid_row,grid_col,query_frames,normalization_key_tokens,pre_softmax,post_softmax,pre_frame_mean,post_frame_mean,pre_pruned,post_pruned';
  const rows = data.pre_softmax.map((value, i) => {
    const cell = state.result.spatial.cells[i];
    return [data.segment_index, data.frame_index, data.layer, i, data.frame_token_start+i, cell?.row ?? '', cell?.col ?? '', data.query_count, data.segment_tokens,
      value, data.post_softmax[i], pre.threshold, post.threshold, Number(pre.pruned[i]), Number(post.pruned[i])].join(',');
  });
  return header+'\n'+rows.join('\n');
}
$('exportFrameQueryCsv').addEventListener('click', () => {
  const data = frameQueryState.data; if (!data) return;
  const url = URL.createObjectURL(new Blob([frameQueryCsv(data)], {type: 'text/csv;charset=utf-8'}));
  const link = node('a'); link.href = url;
  link.download = `frame-query-segment-${data.segment_index+1}-frame-${data.frame_index+1}-layer-${data.layer}.csv`;
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
for (const field of ['Segment', 'Frame', 'Layer']) {
  $('frameQuery'+field).addEventListener('change', () => {
    if (field === 'Segment') frameQueryFrames();
    renderFrameQuery(); syncAttentionSelection('frame');
  });
  $('segmentAttention'+field).addEventListener('change', () => syncAttentionSelection('summary'));
}
$('segmentAttentionSummary').addEventListener('change', () => syncAttentionSelection('summary'));
document.addEventListener('observation:clear', clearFrameQuery);
document.addEventListener('observation:result', initFrameQuery);
