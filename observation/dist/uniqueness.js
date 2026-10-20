'use strict';

window.uniquenessViewer = (() => {
  const el = id => document.getElementById(id);
  const svgNode = (tag, attributes) => {
    const item = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attributes).forEach(([key, value]) => item.setAttribute(key, value));
    return item;
  };
  const number = value => value == null ? '未定义' : value === 0 ? '0' :
    Math.abs(value) < .0001 ? value.toExponential(4) : value.toFixed(6);
  const reasons = {no_other_tokens:'没有其他 token 可比较'};
  let result = null, frameIndex = 0, data = null, token = 0, controller = null, serial = 0, cachedSelection = '', baseURL = '', markers = null;
  const state = () => el('uniquenessState').value;
  const position = () => el('uniquenessPosition').value;
  function reset() {
    ++serial; controller?.abort(); controller = null;
    result = data = markers = null; frameIndex = token = 0; cachedSelection = baseURL = '';
    for (const id of ['uniquenessSection','uniquenessMaps','uniquenessReadout','uniquenessDownloadPlot','uniquenessDownloadData','uniquenessLegend']) el(id).hidden = true;
    el('uniquenessImage').removeAttribute('src'); el('uniquenessOverlay').replaceChildren(); el('uniquenessReadout').textContent = '';
    for (const id of ['uniquenessDownloadPlot','uniquenessDownloadData']) el(id).removeAttribute('href');
  }
  function options(id, count, label) {
    el(id).replaceChildren(...Array.from({length:count},(_,i)=>{
      const option=document.createElement('option');option.value=i;option.textContent=`${label} ${i}`;return option;
    }));
  }
  function configure(value) {
    reset(); result=value; el('uniquenessSection').hidden=false;
    el('uniquenessControls').hidden=!value.kv_snapshot;
    if (!value.kv_snapshot) { el('uniquenessStatus').textContent='旧结果没有 KV 快照；请重新生成观察数据。'; return; }
    options('uniquenessLayer',value.kv_snapshot.layers,'Layer');options('uniquenessHead',value.kv_snapshot.kv_heads,'KV head');
    refresh();
  }
  function selectFrame(index, localToken=0) {
    if (!result?.kv_snapshot) return;
    token=localToken;
    if (index===frameIndex) { updateToken(); return; }
    frameIndex=index;refresh();
  }
  function highlightToken(frame,index) { if(frame===frameIndex){token=index;updateToken();} }
  async function refresh() {
    if (!result?.kv_snapshot) return;
    const layer=Number(el('uniquenessLayer').value),head=Number(el('uniquenessHead').value);
    const base=`/api/uniqueness/${result.job_id}/${layer}/${frameIndex}/${head}`;
    const stage=position(), kind=state(), selection=`${base}:${stage}:${kind}`;
    if (data&&selection===cachedSelection) return;
    const generation=++serial;controller?.abort();controller=new AbortController();data=null;cachedSelection='';markers=null;
    for (const id of ['uniquenessMaps','uniquenessReadout','uniquenessDownloadPlot','uniquenessDownloadData','uniquenessLegend']) el(id).hidden=true;
    el('uniquenessStatus').classList.remove('kv-error');
    el('uniquenessStatus').textContent=`FRAME ${frameIndex+1} · Layer ${layer} · KV head ${head} · ${stage}-RoPE：正在计算 ${kind} 最近邻距离…`;
    try {
      const response=await fetch(`${base}/data.json?position=${stage}&state=${kind}`,{signal:controller.signal});
      if(!response.ok){let message=`HTTP ${response.status}`;try{message=(await response.json()).error||message;}catch{}throw new Error(message);}
      const fetched=await response.json();if(generation!==serial)return;
      data=fetched;cachedSelection=selection;baseURL=base;token=Math.max(0,Math.min(token,data.tokens-1));
      const frame=result.frames[frameIndex];el('uniquenessImage').src=frame.image_url;el('uniquenessImage').alt=`第 ${frameIndex+1} 帧的 ${data.key_position} ${kind} 最近邻距离 叠加图`;
      el('uniquenessOverlay').setAttribute('viewBox',`0 0 ${frame.width} ${frame.height}`);
      el('uniquenessFrameLabel').textContent=`FRAME ${frameIndex+1} · ${frame.timestamp.toFixed(2)} 秒 · ${data.state} · ${data.key_position}`;
      el('uniquenessStatus').textContent=`FRAME ${frameIndex+1} · Layer ${layer} · KV head ${head} · ${data.rows} × ${data.cols} token 网格 · ${data.state} · Key ${data.key_position}`
        + (data.undefined_reason?` · 分数未定义：${reasons[data.undefined_reason]||data.undefined_reason}`:'');
      for (const id of ['uniquenessMaps','uniquenessReadout','uniquenessLegend']) el(id).hidden=false;
      paint();
      el('uniquenessDownloadData').href=`${base}/tensor.npz?position=${stage}&state=${kind}`;el('uniquenessDownloadData').hidden=false;
    } catch(error) {
      if(generation!==serial||error.name==='AbortError')return;
      el('uniquenessStatus').textContent=`Uniqueness 加载失败：${error.message}`;el('uniquenessStatus').classList.add('kv-error');
    }
  }
  function cap() { return el('uniquenessScale').value==='frame'&&data.max_uniqueness>0?data.max_uniqueness:2; }
  function rgb(value) {
    if(value==null)return [213,220,224];
    const t=Math.max(0,Math.min(1,value/cap()));
    return [255,247,236].map((v,i)=>Math.round(v+([190,55,15][i]-v)*t));
  }
  function paint() {
    if(!data)return;
    const canvas=el('uniquenessCanvas');canvas.width=data.cols;canvas.height=data.rows;
    const pixels=canvas.getContext('2d').createImageData(data.cols,data.rows);
    data.uniqueness.forEach((value,i)=>pixels.data.set([...rgb(value),255],i*4));
    canvas.getContext('2d').putImageData(pixels,0,0);
    el('uniquenessCursor').setAttribute('viewBox',`0 0 ${data.cols} ${data.rows}`);
    el('uniquenessScaleMax').textContent=number(cap());
    el('uniquenessGridLabel').textContent=`${data.rows} × ${data.cols} · 每格一个 token · 行、列从 0 开始`;
    const overlay=el('uniquenessOverlay'),frame=result.frames[frameIndex],cells=svgNode('g',{});
    overlay.replaceChildren(cells);markers=svgNode('g',{'pointer-events':'none'});overlay.append(markers);
    for(const cell of result.spatial?.cells||[]) {
      const box=cell.box,rect=svgNode('rect',{x:box[0]*frame.width,y:box[1]*frame.height,
        width:(box[2]-box[0])*frame.width,height:(box[3]-box[1])*frame.height,
        fill:`rgb(${rgb(data.uniqueness[cell.index]).join(',')})`,'fill-opacity':el('uniquenessOpacity').value,
        'data-token':cell.index});
      rect.addEventListener('mouseenter',()=>{token=cell.index;updateToken();});
      rect.addEventListener('click',()=>window.selectFrame(frameIndex,false,cell.index));cells.append(rect);
    }
    el('uniquenessDownloadPlot').href=`${baseURL}/map.png?scale=${el('uniquenessScale').value}&position=${position()}&state=${state()}`;
    el('uniquenessDownloadPlot').download=`uniqueness-${state()}-${position()}-layer${data.layer}-frame${data.frame}-head${data.head}.png`;
    el('uniquenessDownloadPlot').hidden=false;
    updateToken();
  }
  function updateToken() {
    if(!data)return;
    token=Math.max(0,Math.min(data.tokens-1,token));
    const nearest=data.nearest_token[token],cursor=el('uniquenessCursor');cursor.replaceChildren();markers?.replaceChildren();
    const frame=result.frames[frameIndex];
    for(const [index,stroke,dash] of [[token,'#087f80',''],[nearest,'#763ab5','3 2']]) {
      if(index==null)continue;
      cursor.append(svgNode('rect',{x:index%data.cols,y:Math.floor(index/data.cols),width:1,height:1,
        fill:'none',stroke,'stroke-width':.1,'stroke-dasharray':dash?'.18 .1':''}));
      const box=result.spatial?.cells[index]?.box;
      if(box&&markers)markers.append(svgNode('rect',{x:box[0]*frame.width,y:box[1]*frame.height,width:(box[2]-box[0])*frame.width,
        height:(box[3]-box[1])*frame.height,fill:'none',stroke,'stroke-width':2,'stroke-dasharray':dash,'vector-effect':'non-scaling-stroke'}));
    }
    el('uniquenessReadout').textContent=`${data.state} · ${data.key_position} · token i = ${token}（全局 #${data.token_start+token} · 网格 ${Math.floor(token/data.cols)}, ${token%data.cols}）\n`
      + `dᵢᴺᴺ = ${number(data.uniqueness[token])} · 最近似其他 token j = ${nearest==null?'未定义':nearest}\n`
      + `τ = mean(dᴺᴺ) = ${number(data.tau)} · dᵢᴺᴺ < τ：${data.redundant[token]==null?'未定义':data.redundant[token]?'是':'否'} · R_NN = ${number(data.score)} · 平均 dᴺᴺ = ${number(data.mean_uniqueness)}`;
  }
  const canvas=el('uniquenessCanvas');
  function pick(event) {
    if(!data)return;
    const b=canvas.getBoundingClientRect(),col=Math.max(0,Math.min(data.cols-1,Math.floor((event.clientX-b.left)/b.width*data.cols))),
      row=Math.max(0,Math.min(data.rows-1,Math.floor((event.clientY-b.top)/b.height*data.rows)));
    token=row*data.cols+col;updateToken();
  }
  canvas.addEventListener('mousemove',pick);
  canvas.addEventListener('click',event=>{pick(event);if(data)window.selectFrame(frameIndex,false,token);});
  canvas.addEventListener('keydown',event=>{
    if(!data)return;const moves={ArrowLeft:-1,ArrowRight:1,ArrowUp:-data.cols,ArrowDown:data.cols};if(!(event.key in moves))return;
    event.preventDefault();token=Math.max(0,Math.min(data.tokens-1,token+moves[event.key]));updateToken();window.selectFrame(frameIndex,false,token);
  });
  ['uniquenessLayer','uniquenessHead'].forEach(id=>el(id).addEventListener('change',refresh));
  el('uniquenessScale').addEventListener('change',paint);el('uniquenessOpacity').addEventListener('input',paint);
  return {reset,configure,selectFrame,highlightToken,setPosition:refresh};
})();
