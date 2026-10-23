'use strict';
const segmentAttentionState = {controller: null, request: 0, data: null, selected: 0};
const attentionMeasures = [
  {key: 'pre_softmax', label: 'Softmax 前 · 平均 QKᵀ / √d'},
  {key: 'post_softmax', label: 'Softmax 后 · 平均注意力权重'},
];
const ropeLabel = mode => mode === 'post' ? 'RoPE 后' : 'RoPE 前';
function clearSegmentAttention() {
  segmentAttentionState.controller?.abort(); segmentAttentionState.request++;
  segmentAttentionState.data = null;
  $('segmentAttentionPanel').hidden = true; $('exportAttentionCsv').disabled = true;
  $('segmentAttentionMaps').replaceChildren();
}
function attentionFrames() {
  const segment = state.result.segment_experiment.segments[Number($('segmentAttentionSegment').value)];
  $('segmentAttentionFrame').replaceChildren();
  segment.frame_indices.forEach((index, local) => {
    const frame = state.result.frames[index], option = node('option', '', `段内 ${local + 1}/${segment.frame_indices.length} · 帧 ${index + 1} · ${fmt(frame.timestamp, 2)}s`);
    option.value = index; $('segmentAttentionFrame').append(option);
  });
}
function initSegmentAttention() {
  clearSegmentAttention();
  const experiment = state.result?.segment_experiment; if (!experiment) return;
  $('segmentAttentionPanel').hidden = false;
  const modes = experiment.segment_attention?.rope_modes || ['pre'];
  $('segmentAttentionRope').replaceChildren();
  for (const mode of modes.includes('post') ? ['both', 'pre', 'post'] : ['pre']) {
    const option = node('option', '', mode === 'both' ? 'RoPE 前后对比' : ropeLabel(mode));
    option.value = mode; $('segmentAttentionRope').append(option);
  }
  if (!experiment.segment_attention?.url) {
    $('segmentAttentionMeta').textContent = '旧缓存没有 summary Query，请重启服务并重新生成观察数据。';
    for (const id of ['segmentAttentionRope', 'segmentAttentionLayer', 'segmentAttentionSummary', 'segmentAttentionSegment', 'segmentAttentionFrame']) { $(id).replaceChildren(); $(id).disabled = true; }
    $('segmentAttentionSummaryText').textContent = ''; $('segmentAttentionMass').textContent = '';
    return;
  }
  for (const id of ['segmentAttentionRope', 'segmentAttentionLayer', 'segmentAttentionSummary', 'segmentAttentionSegment', 'segmentAttentionFrame']) $(id).disabled = false;
  $('segmentAttentionLayer').replaceChildren(node('option', '', '全部层平均'));
  $('segmentAttentionLayer').options[0].value = 'all';
  for (let layer = 0; layer < state.result.model.layers; layer++) {
    const option = node('option', '', `第 ${layer + 1} 层`); option.value = layer; $('segmentAttentionLayer').append(option);
  }
  for (const id of ['segmentAttentionSummary', 'segmentAttentionSegment']) {
    $(id).replaceChildren();
    experiment.segments.forEach(segment => {
      const option = node('option', '', `${id === 'segmentAttentionSummary' ? 'Summary' : 'Segment'} ${segment.id + 1} · ${fmt(segment.first_timestamp, 1)}–${fmt(segment.last_timestamp, 1)}s`);
      option.value = segment.id; $(id).append(option);
    });
  }
  segmentAttentionState.selected = 0; attentionFrames(); renderSegmentAttention();
}
async function renderSegmentAttention() {
  segmentAttentionState.controller?.abort(); const request = ++segmentAttentionState.request;
  segmentAttentionState.data = null; $('exportAttentionCsv').disabled = true;
  $('segmentAttentionMaps').replaceChildren(); $('segmentAttentionMass').textContent = '';
  $('segmentAttentionPositions').textContent = '';
  const experiment = state.result?.segment_experiment;
  if (!experiment?.segment_attention?.url) return;
  const summary = Number($('segmentAttentionSummary').value), segment = Number($('segmentAttentionSegment').value);
  const frame = Number($('segmentAttentionFrame').value), layer = $('segmentAttentionLayer').value;
  const rope = $('segmentAttentionRope').value;
  $('segmentAttentionSummaryText').textContent = experiment.segments[summary].summary.text || '（无内容 token。）';
  $('segmentAttentionMeta').textContent = '正在计算整个 segment 上的 QK 分数及 softmax 权重…';
  $('segmentAttentionReadout').textContent = '正在读取…';
  const controller = new AbortController(); segmentAttentionState.controller = controller;
  try {
    const response = await fetch(`${experiment.segment_attention.url}?${new URLSearchParams({summary, segment, frame, layer, rope})}`, {signal: controller.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    if (request !== segmentAttentionState.request) return;
    segmentAttentionState.data = data;
    $('segmentAttentionMeta').textContent = `${rope === 'both' ? 'RoPE 前后对比' : ropeLabel(rope)} · ${layer === 'all' ? '全部层平均' : `第 ${Number(layer) + 1} 层`} · ${data.text_tokens.length} 个 summary Query · ${data.segment_tokens} 个 segment Key · mask ${data.masked_summary_tokens} 个 summary Key`;
    $('segmentAttentionMass').textContent = data.variants.map(v => `${ropeLabel(v.rope)}：当前帧权重和 ${fmt(v.frame_masses.find(f => f.frame_index === frame)?.mass, 6)} · segment 权重和 ${fmt(v.segment_mass, 6)} · 被遮蔽 summary 权重和 ${fmt(v.summary_mass, 6)}`).join('\n');
    const positions = data.positions;
    $('segmentAttentionPositions').textContent = `原生 token 位置：segment [${positions.video_token_start}, ${positions.video_token_end})；独立 summary [${positions.summary_token_start ?? '—'}, ${positions.summary_token_end ?? '—'})。`;
    $('exportAttentionCsv').disabled = !data.text_tokens.length;
    for (const variant of data.variants) for (const measure of attentionMeasures) renderAttentionMap(data, measure, variant);
    readAttentionToken(segmentAttentionState.selected);
  } catch (error) {
    if (error.name === 'AbortError' || request !== segmentAttentionState.request) return;
    $('segmentAttentionMeta').textContent = `注意力读取失败：${error.message}`;
    $('segmentAttentionReadout').textContent = '可重新选择帧或层重试。';
  }
}
function renderAttentionMap(data, measure, variant) {
  const frame = state.result.frames[data.frame_index], values = variant[measure.key];
  const label = `${ropeLabel(variant.rope)} · ${measure.label}`;
  const range = similarityColorRange(values), pruning = frameMeanPruning(values);
  const section = node('div', 'attention-measure'); section.dataset.measure = measure.key;
  section.dataset.rope = variant.rope;
  section.append(node('h3', '', label));
  const legend = node('div', 'cosine-legend'), ramp = node('span', 'cosine-ramp');
  ramp.style.background = range.min == null ? '#dce2e5' : range.min === range.max ? cosineColor(range.min, range) : 'linear-gradient(90deg,white,rgb(8,127,128))';
  legend.append(node('span', '', `min ${fmt(range.min, 6)}`), ramp, node('span', '', `max ${fmt(range.max, 6)}`), node('span', '', '当前帧实际范围'));
  section.append(legend);
  const pair = node('div', 'similarity-image-pair');
  for (const masked of [false, true]) {
    const figure = node('figure'); figure.append(node('figcaption', '', masked ? '剪枝预览 · 低于当前帧均值留白' : label));
    const stage = node('div', 'attention-image-stage'); stage.style.aspectRatio = `${frame.width} / ${frame.height}`;
    const img = node('img'); img.src = frame.image_url; img.alt = `帧 ${data.frame_index + 1} · ${label}${masked ? '剪枝预览' : '热力图'}`;
    const svg = svgNode('svg', {viewBox: '0 0 1000 1000', preserveAspectRatio: 'none', role: 'img', 'aria-label': img.alt});
    svg.classList.add(masked ? 'attention-pruned-mask' : 'attention-overlay');
    if (!masked) svg.setAttribute('tabindex', 0);
    state.result.spatial.cells.forEach((cell, index) => {
      if (masked && !pruning.pruned[index]) return;
      const [x1,y1,x2,y2] = cell.box;
      const rect = svgNode('rect', {x:x1*1000,y:y1*1000,width:(x2-x1)*1000,height:(y2-y1)*1000,
        fill:masked?'#fff':cosineColor(values[index],range),'fill-opacity':masked?1:.5,
        'data-token-index':index,stroke:masked?'none':'#ffffff55','stroke-width':1});
      rect.append(svgNode('title',{},`图片 token #${index} · ${label}: ${fmt(values[index],8)}`));
      if (!masked) {
        rect.addEventListener('mouseenter',()=>readAttentionToken(index));
        rect.addEventListener('click',()=>{segmentAttentionState.selected=index;readAttentionToken(index);});
      }
      svg.append(rect);
    });
    if (!masked) {
      svg.addEventListener('mouseleave',()=>readAttentionToken(segmentAttentionState.selected));
      svg.addEventListener('keydown',event=>{
        const cols=state.result.spatial.cols, moves={ArrowLeft:-1,ArrowRight:1,ArrowUp:-cols,ArrowDown:cols};
        if (!(event.key in moves)) return;
        event.preventDefault();segmentAttentionState.selected=Math.max(0,Math.min(data.image_tokens-1,segmentAttentionState.selected+moves[event.key]));readAttentionToken(segmentAttentionState.selected);
      });
    }
    stage.append(img,svg);figure.append(stage);
    if (masked && pruning.threshold==null) figure.hidden=true;
    pair.append(figure);
  }
  section.append(pair,node('p','muted',pruning.threshold==null?'无有效 Query，无法计算分数或剪枝阈值。':`帧内均值 ${fmt(pruning.threshold,8)} · 剪去 ${pruning.pruned.filter(Boolean).length}/${data.image_tokens} 个 token；等于均值保留。`));
  $('segmentAttentionMaps').append(section);
}
function readAttentionToken(index) {
  const data=segmentAttentionState.data;if(!data)return;
  const cell=state.result.spatial.cells[index];
  $('segmentAttentionReadout').textContent=`Summary ${data.summary_index+1} → Segment ${data.segment_index+1} · 帧 ${data.frame_index+1} · 图片 token #${index} · 全局 token #${data.frame_token_start+index}${cell?` · 空间行 ${cell.row+1} / 列 ${cell.col+1}`:''}\n` + data.variants.map(v => `${ropeLabel(v.rope)}：Softmax 前 ${fmt(v.pre_softmax[index],8)} · Softmax 后 ${fmt(v.post_softmax[index],8)}`).join('\n');
  document.querySelectorAll('.attention-overlay rect').forEach(rect=>{
    const selected=Number(rect.dataset.tokenIndex)===index;rect.setAttribute('stroke',selected?'#ff8a24':'#ffffff55');rect.setAttribute('stroke-width',selected?5:1);
  });
}
$('segmentAttentionSummary').addEventListener('change',()=>{$('segmentAttentionSegment').value=$('segmentAttentionSummary').value;attentionFrames();renderSegmentAttention();});
$('segmentAttentionSegment').addEventListener('change',()=>{attentionFrames();renderSegmentAttention();});
$('segmentAttentionFrame').addEventListener('change',renderSegmentAttention);
$('segmentAttentionLayer').addEventListener('change',renderSegmentAttention);
$('segmentAttentionRope').addEventListener('change',renderSegmentAttention);
$('exportAttentionCsv').addEventListener('click',()=>{
  const data=segmentAttentionState.data;if(!data?.text_tokens.length)return;
  const fields=['pre_softmax','post_softmax','pre_frame_mean','post_frame_mean','pre_pruned','post_pruned','summary_mass'];
  const header=['summary_index','segment_index','frame_index','layer','image_token','global_token','grid_row','grid_col','normalization_key_tokens','masked_summary_tokens',...data.variants.flatMap(v=>fields.map(f=>`${v.rope}_rope_${f}`))].join(',');
  const stats=data.variants.map(v=>({variant:v,pre:frameMeanPruning(v.pre_softmax),post:frameMeanPruning(v.post_softmax)}));
  const rows=Array.from({length:data.image_tokens},(_,j)=>{const cell=state.result.spatial.cells[j];return [data.summary_index,data.segment_index,data.frame_index,data.layer,j,data.frame_token_start+j,cell?.row??'',cell?.col??'',data.segment_tokens,data.masked_summary_tokens,...stats.flatMap(({variant:v,pre,post})=>[v.pre_softmax[j]??'',v.post_softmax[j]??'',pre.threshold??'',post.threshold??'',Number(pre.pruned[j]),Number(post.pruned[j]),v.summary_mass??''])].join(',');});
  const url=URL.createObjectURL(new Blob([header+'\n'+rows.join('\n')],{type:'text/csv;charset=utf-8'}));
  const link=node('a');link.href=url;link.download=`summary-${data.summary_index+1}-segment-${data.segment_index+1}-frame-${data.frame_index+1}-layer-${data.layer}-${data.requested_rope}-rope-attention.csv`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
document.addEventListener('observation:clear',clearSegmentAttention);
document.addEventListener('observation:result',initSegmentAttention);
