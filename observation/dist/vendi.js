'use strict';

window.vendiViewer = (() => {
  const el = id => document.getElementById(id);
  const make = (tag, attrs = {}, text) => {
    const item = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attrs).forEach(([key, value]) => item.setAttribute(key, value));
    if (text !== undefined) item.textContent = text;
    return item;
  };
  const format = value => value == null ? '未定义' :
    value !== 0 && Math.abs(value) < .0001 ? value.toExponential(4) : value.toFixed(4);
  const quantile = (a, p) => { const i = (a.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i); return a[lo] + (a[hi] - a[lo]) * (i - lo); };
  function summarize(values) {
    const a = values.filter(Number.isFinite).sort((x, y) => x - y);
    return {count:a.length, total:values.length, min:a[0], max:a.at(-1),
      mean:a.length ? a.reduce((x,y) => x+y, 0) / a.length : null,
      median:a.length ? quantile(a,.5) : null, q1:a.length ? quantile(a,.25) : null, q3:a.length ? quantile(a,.75) : null};
  }
  let result=null, selectedFrame=0, statistics=[];
  const state = () => el('uniquenessState').value;
  const position = () => el('uniquenessPosition').value;
  const metric = () => position()==='pre' ? result?.uniqueness_pre_rope : result?.uniqueness;
  function reset() {
    result=null; selectedFrame=0; statistics=[];
    el('vendiPanel').hidden=true; el('vendiChart').replaceChildren(); el('vendiReadout').textContent='';
    for (const id of ['vendiDownloadPlot','vendiDownloadData']) { el(id).hidden=true; el(id).removeAttribute('href'); }
    el('exportVendiCsv').disabled=true;
  }
  function option(value,text) { const item=document.createElement('option'); item.value=value; item.textContent=text; return item; }
  function configure(value) {
    reset(); result=value; el('vendiPanel').hidden=false;
    el('uniquenessPreOption').disabled=!value.pre_rope_key_snapshot;
    if (!value.pre_rope_key_snapshot) el('uniquenessPosition').value='post';
    el('uniquenessPositionHint').textContent=value.pre_rope_key_snapshot
      ? 'RoPE 前直接采集 k_proj 输出；RoPE 后读取原生 KV cache。K 按所选 RoPE 阶段计算，V 使用原生缓存且两阶段相同。切换同时更新分数图和空间热图。'
      : '该结果未保存 RoPE 前 Key；重新生成观察数据后可切换。';
    const m=metric(), available=m?.metric==='within_frame_separate_kv_nearest_neighbor_redundancy_mean_threshold';
    for (const id of ['vendiControls','vendiLegend','vendiReadout']) el(id).hidden=!available;
    if (!available) { el('vendiStatus').textContent='旧结果不含 K/V 最近邻冗余比例 R_NN；请重新生成观察数据。已有 KV 快照可离线计算。'; return; }
    for (const [id,count,label] of [['vendiLayer',m.layers,'Layer'],['vendiHead',m.kv_heads,'KV head']]) {
      el(id).replaceChildren(option('all',id==='vendiLayer'?'全部层':'全部 KV heads'), ...Array.from({length:count},(_,i)=>option(i,`${label} ${i}`)));
    }
    el('vendiFrame').replaceChildren(...value.frames.map((f,i)=>option(i,`FRAME ${i+1} · ${f.timestamp.toFixed(2)}s`)));
    el('exportVendiCsv').disabled=false; render();
  }
  function values(field, frame, source=metric()) {
    const layer=el('vendiLayer').value, head=el('vendiHead').value;
    return source[field][state()].flatMap((frames,l)=>layer!=='all'&&l!==Number(layer)?[]:
      frames[frame].filter((_,h)=>head==='all'||h===Number(head)));
  }
  function selectFrame(index) { if (!statistics.length) return; selectedFrame=index; updateFrame(); }
  function updateFrame() {
    if (!statistics[selectedFrame]) return;
    el('vendiFrame').value=selectedFrame;
    el('vendiChart').querySelectorAll('[data-frame]').forEach(item=>{
      const active=Number(item.dataset.frame)===selectedFrame; item.classList.toggle('selected',active); item.setAttribute('aria-pressed',String(active));
    });
    const s=statistics[selectedFrame];
    const m=metric();
    el('vendiReadout').textContent=`FRAME ${selectedFrame+1} · ${result.frames[selectedFrame].timestamp.toFixed(2)} 秒 · ${m.key_position}\n`
      + `${state()} 最近邻冗余比例 R_NN：均值 ${format(s.mean)} · 中位数 ${format(s.median)} · Q1–Q3 ${format(s.q1)}–${format(s.q3)} · 范围 ${format(s.min)}–${format(s.max)} · 有效 ${s.count}/${s.total}\n`
      + `所选层/head 的平均 τ = ${format(summarize(values('thresholds',selectedFrame)).mean)} · 平均 Σdᴺᴺ = ${format(summarize(values('uniqueness_sums',selectedFrame)).mean)} · 平均最近邻距离 = ${format(summarize(values('mean_uniqueness',selectedFrame)).mean)}`;
    if (result.uniqueness_pre_rope?.metric==='within_frame_separate_kv_nearest_neighbor_redundancy_mean_threshold' && result.uniqueness?.metric==='within_frame_separate_kv_nearest_neighbor_redundancy_mean_threshold') {
      const pre=summarize(values('scores',selectedFrame,result.uniqueness_pre_rope));
      const post=summarize(values('scores',selectedFrame,result.uniqueness));
      el('vendiReadout').textContent+=`\n同一筛选下的 R_NN 均值：RoPE 前 ${format(pre.mean)}（${pre.count}/${pre.total}） · RoPE 后 ${format(post.mean)}（${post.count}/${post.total}）`;
    }
    const layer=el('vendiLayer').value, head=el('vendiHead').value;
    el('vendiDownloadPlot').href=`/api/uniqueness/${result.job_id}/distribution.png?`+new URLSearchParams({layer,head,frame:selectedFrame,position:position(),state:state()});
    el('vendiDownloadPlot').download=`uniqueness-${state()}-${position()}-redundancy.png`;
    el('vendiDownloadPlot').hidden=false;
    el('vendiDownloadData').hidden=layer==='all'||head==='all';
    if (!el('vendiDownloadData').hidden) el('vendiDownloadData').href=`/api/uniqueness/${result.job_id}/${layer}/${selectedFrame}/${head}/tensor.npz?position=${position()}&state=${state()}`;
  }
  function render() {
    const m=metric();
    if (m?.metric!=='within_frame_separate_kv_nearest_neighbor_redundancy_mean_threshold') {
      statistics=[]; el('vendiChart').replaceChildren(); el('vendiReadout').textContent='';
      el('exportVendiCsv').disabled=true; el('vendiDownloadPlot').hidden=el('vendiDownloadData').hidden=true;
      el('vendiStatus').textContent='当前结果不含所选阶段的最近邻冗余比例，请重新生成观察数据。';
      return;
    }
    el('exportVendiCsv').disabled=false;
    statistics=result.frames.map((_,i)=>summarize(values('scores',i)));
    const width=Math.max(820,m.frames*75+90), height=320, left=58, right=width-22, top=32, bottom=247;
    const step=(right-left)/m.frames, x=i=>left+(i+.5)*step;
    const maximum=1, y=v=>bottom-v/maximum*(bottom-top), color='#087f80';
    const svg=make('svg',{viewBox:`0 0 ${width} ${height}`,role:'group','aria-label':'逐帧 K/V 最近邻冗余比例 R_NN 分布',style:`min-width:${width}px`});
    svg.append(make('text',{x:left,y:15},`${state()} 最近邻冗余比例 R_NN · ${m.key_position}`));
    for (let i=0;i<=4;i++) { const v=maximum*i/4; svg.append(make('line',{x1:left,x2:right,y1:y(v),y2:y(v),stroke:'#e2e9eb'}),make('text',{x:left-9,y:y(v)+4,'text-anchor':'end'},v.toFixed(2))); }
    statistics.forEach((s,i)=>{
      const center=x(i), w=Math.min(24,step*.4);
      if (s.count) {
        svg.append(make('line',{x1:center,x2:center,y1:y(s.min),y2:y(s.max),stroke:color}),
          make('rect',{x:center-w/2,y:y(s.q3),width:w,height:Math.max(.8,y(s.q1)-y(s.q3)),fill:color,'fill-opacity':.2,stroke:color}),
          make('line',{x1:center-w/2,x2:center+w/2,y1:y(s.median),y2:y(s.median),stroke:color,'stroke-width':2}),
          make('circle',{cx:center,cy:y(s.mean),r:3,fill:color}));
        for (const v of [s.min,s.max]) svg.append(make('line',{x1:center-w/3,x2:center+w/3,y1:y(v),y2:y(v),stroke:color}));
      } else svg.append(make('text',{x:center,y:bottom-6,'text-anchor':'middle'},'×'));
      svg.append(make('text',{x:center,y:bottom+22,'text-anchor':'middle'},`${i+1}`),make('text',{x:center,y:bottom+39,'text-anchor':'middle'},`${result.frames[i].timestamp.toFixed(2)}s`));
      const hit=make('rect',{x:center-step/2+2,y:top-7,width:step-4,height:bottom-top+14,fill:'transparent',class:'vendi-hit','data-frame':i,role:'button',tabindex:0,'aria-label':`第 ${i+1} 帧，分数均值 ${format(s.mean)}`});
      hit.append(make('title',{},`${state()} 最近邻冗余比例 R_NN ${format(s.mean)}`));
      hit.addEventListener('click',()=>window.selectFrame(i));
      hit.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();window.selectFrame(i);}});
      svg.append(hit);
    });
    svg.append(make('text',{x:(left+right)/2,y:height-7,'text-anchor':'middle'},'采样帧 / 源视频时间'));
    el('vendiChart').replaceChildren(svg);
    const missing=statistics.reduce((sum,s)=>sum+s.total-s.count,0);
    el('vendiStatus').textContent=`${state()} · ${m.key_position} · ${m.frames} 帧 · 同帧 ${m.tokens_per_frame} 个 K/V token，排除自身比较 · 每帧 ${statistics[0].total} 个层/head 分数 · τ = 各切片 dᴺᴺ 均值 · 范围 0–1，表示低于本切片均值的比例。`
      +(missing?` ${missing} 个分数未定义（无其他 token 可比较）。`:'');
    updateFrame();
  }
  function exportCsv() {
    const m=metric(); if(!m)return;
    const rows=[['frame_index','source_index','timestamp_seconds','token_start','token_end','layer','kv_head','state','nearest_neighbor_redundancy','uniqueness_sum','mean_uniqueness','max_uniqueness','zero_norm_tokens','undefined_reason','key_position','metric','tau','epsilon']];
    result.frames.forEach((f,i)=>{for(let l=0;l<m.layers;l++)for(let h=0;h<m.kv_heads;h++)for(const kind of ['key','value'])rows.push([i,f.source_index,f.timestamp,f.token_start,f.token_end,l,h,kind,
      ...['scores','uniqueness_sums','mean_uniqueness','max_uniqueness','zero_norm_tokens','undefined_reasons'].map(name=>m[name][kind][l][i][h]),m.key_position,m.metric,m.thresholds[kind][l][i][h],m.epsilon]);});
    const blob=new Blob(['\uFEFF',rows.map(row=>row.join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'});
    const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=`kv-uniqueness-${position()}-${result.job_id}.csv`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  ['vendiLayer','vendiHead'].forEach(id=>el(id).addEventListener('change',render));
  el('vendiFrame').addEventListener('change',()=>{if(metric())window.selectFrame(Number(el('vendiFrame').value));});
  ['uniquenessPosition','uniquenessState'].forEach(id=>el(id).addEventListener('change',()=>{render();window.uniquenessViewer.setPosition();}));
  el('exportVendiCsv').addEventListener('click',exportCsv);
  return {reset,configure,selectFrame};
})();
