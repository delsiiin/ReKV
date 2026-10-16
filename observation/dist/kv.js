'use strict';

// Loaded before app.js. Methods access the main viewer only after it initializes.
window.kvViewer = (() => {
  const el = id => document.getElementById(id);
  let result = null, frameIndex = 0, data = null, serial = 0, controller = null, imageURL = null;
  let token = 0, channel = 0, cachedSelection = '', baseKey = null, baseValue = null;
  const number = value => value === 0 ? '0' : (Math.abs(value) < .001 ? value.toExponential(3) : value.toFixed(4));

  function reset() {
    ++serial;
    controller?.abort(); controller = null;
    if (imageURL) URL.revokeObjectURL(imageURL);
    imageURL = null; result = data = baseKey = baseValue = null; cachedSelection = '';
    ['kvPanel', 'kvSurface', 'kvHeatmaps', 'kvReadout', 'kvDownloadPlot', 'kvDownloadData'].forEach(id => el(id).hidden = true);
    el('kvSurface').removeAttribute('src');
  }
  function options(id, count, label) {
    el(id).replaceChildren(...Array.from({length: count}, (_, index) => {
      const option = document.createElement('option'); option.value = index; option.textContent = label(index); return option;
    }));
  }
  function configure(value) {
    reset(); result = value; frameIndex = token = channel = 0;
    el('kvPanel').hidden = false;
    if (!result.kv_snapshot) { el('kvStatus').textContent = '旧结果尚未保存 K/V；请重新生成观察数据。'; return; }
    options('kvFrame', result.frames.length, i => `FRAME ${i + 1} · ${result.frames[i].timestamp.toFixed(2)}s`);
    options('kvLayer', result.kv_snapshot.layers, i => `Layer ${i}`);
    options('kvHead', result.kv_snapshot.kv_heads, i => `KV head ${i}`);
    refresh();
  }
  function selectFrame(index, localToken = 0) {
    if (!result?.kv_snapshot) return;
    token = localToken;
    if (frameIndex === index) { drawHeatmaps(); return; }
    frameIndex = index; el('kvFrame').value = index; refresh();
  }
  function highlightToken(frame, localToken) {
    if (frameIndex === frame && data && token !== localToken) { token = localToken; drawHeatmaps(); }
  }
  async function checkedFetch(url, signal, kind) {
    const response = await fetch(url, {signal});
    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      try { message = (await response.json()).error || message; } catch {}
      throw new Error(message);
    }
    return kind === 'json' ? response.json() : response.blob();
  }
  async function refresh() {
    if (!result?.kv_snapshot) return;
    const generation = ++serial;
    controller?.abort(); controller = new AbortController();
    const signal = controller.signal;
    const layer = Number(el('kvLayer').value), head = Number(el('kvHead').value);
    const selection = `${result.job_id}/${layer}/${frameIndex}/${head}`;
    const base = `${result.kv_snapshot.api_base}/${layer}/${frameIndex}/${head}`;
    const query = new URLSearchParams({elevation: el('kvElevation').value, azimuth: el('kvAzimuth').value, shared: el('kvSharedScale').checked ? '1' : '0'});
    el('kvStatus').textContent = `FRAME ${frameIndex + 1} · Layer ${layer} · KV head ${head}：正在绘制完整 token × channel 曲面…`;
    el('kvStatus').classList.remove('kv-error');
    el('kvSurface').hidden = true; el('kvDownloadPlot').hidden = true;
    if (selection !== cachedSelection) {
      data = null; el('kvHeatmaps').hidden = true; el('kvReadout').hidden = true;
    }
    el('kvDownloadData').hidden = false; el('kvDownloadData').href = `${base}/tensor.npz`;
    const dataTask = (async () => {
      if (!data || selection !== cachedSelection) {
        const fetched = await checkedFetch(`${base}/data.json`, signal, 'json');
        if (generation !== serial) return;
        data = fetched; cachedSelection = selection;
      }
      if (generation !== serial) return;
      prepareHeatmaps();
      el('kvHeatmaps').hidden = false; el('kvReadout').hidden = false;
    })();
    const figureTask = (async () => {
      const blob = await checkedFetch(`${base}/surface.png?${query}`, signal, 'blob');
      if (generation !== serial) return;
      if (imageURL) URL.revokeObjectURL(imageURL);
      imageURL = URL.createObjectURL(blob);
      el('kvSurface').src = imageURL;
      el('kvSurface').hidden = false;
      el('kvDownloadPlot').href = `${base}/surface.png?${query}`;
      el('kvDownloadPlot').download = `kv-frame${frameIndex + 1}-layer${layer}-head${head}.png`;
      el('kvDownloadPlot').hidden = false;
    })();
    const outcomes = await Promise.allSettled([dataTask, figureTask]);
    if (generation !== serial) return;
    const failure = outcomes.find(outcome => outcome.status === 'rejected');
    if (failure) {
      if (failure.reason.name !== 'AbortError') {
        el('kvStatus').textContent = `K/V 视图加载失败：${failure.reason.message}`;
        el('kvStatus').classList.add('kv-error');
      }
      return;
    }
    el('kvStatus').textContent = `FRAME ${frameIndex + 1} · Layer ${layer} · KV head ${head} · ${data.tokens} token × ${data.channels} channel · Key: post-RoPE · Value: 原始缓存 · ${el('kvSharedScale').checked ? 'K/V 共用标尺' : 'K/V 各自标尺'}`;
  }
  function rgb(t) {
    // Blue -> near white -> red, matching the surface's coolwarm palette.
    const low = [59, 76, 192], mid = [221, 221, 221], high = [180, 4, 38];
    const a = t < .5 ? low : mid, b = t < .5 ? mid : high, f = t < .5 ? t * 2 : (t - .5) * 2;
    return a.map((v, i) => Math.round(v + (b[i] - v) * f));
  }
  function prepareHeatmaps() {
    const shared = el('kvSharedScale').checked ? Math.max(data.key.maximum, data.value.maximum) : null;
    for (const kind of ['key', 'value']) {
      const cap = kind === 'key' ? 'Key' : 'Value', tensor = data[kind];
      const canvas = el(`kv${cap}Heatmap`), max = shared ?? tensor.maximum;
      canvas.width = data.tokens; canvas.height = data.channels;
      const pixels = canvas.getContext('2d').createImageData(data.tokens, data.channels);
      for (let c = 0; c < data.channels; c++) for (let t = 0; t < data.tokens; t++) {
        const color = rgb(max > 0 ? tensor.magnitude[t][c] / max : 0), offset = (c * data.tokens + t) * 4;
        pixels.data.set([...color, 255], offset);
      }
      if (kind === 'key') baseKey = pixels; else baseValue = pixels;
      el(`kv${cap}Scale`).textContent = `0 → ${number(max)}`;
      el(`kv${cap}Axis`).textContent = `Token 0 → ${data.tokens - 1} · Channel 0 → ${data.channels - 1}`;
      el(`kv${cap}Stats`).textContent = `max ${number(tensor.maximum)} · mean |${cap[0]}| ${number(tensor.mean)} · P99 ${number(tensor.p99)}`;
    }
    drawHeatmaps();
  }
  function drawHeatmaps() {
    if (!data || !baseKey || !baseValue) return;
    token = Math.max(0, Math.min(data.tokens - 1, token)); channel = Math.max(0, Math.min(data.channels - 1, channel));
    for (const [id, pixels] of [['kvKeyHeatmap', baseKey], ['kvValueHeatmap', baseValue]]) {
      const ctx = el(id).getContext('2d'); ctx.putImageData(pixels, 0, 0);
      ctx.strokeStyle = 'rgba(0,0,0,.55)'; ctx.lineWidth = .6; ctx.beginPath();
      ctx.moveTo(token + .5, 0); ctx.lineTo(token + .5, data.channels);
      ctx.moveTo(0, channel + .5); ctx.lineTo(data.tokens, channel + .5); ctx.stroke();
    }
    el('kvReadout').textContent = `FRAME ${frameIndex + 1} · token #${data.token_start + token}（帧内 ${token}） · channel ${channel} · |K| = ${number(data.key.magnitude[token][channel])} · |V| = ${number(data.value.magnitude[token][channel])}`;
  }
  for (const id of ['kvKeyHeatmap', 'kvValueHeatmap']) {
    const canvas = el(id);
    const pick = event => {
      if (!data) return;
      const bounds = canvas.getBoundingClientRect();
      token = Math.floor((event.clientX - bounds.left) / bounds.width * data.tokens);
      channel = Math.floor((event.clientY - bounds.top) / bounds.height * data.channels);
      drawHeatmaps();
    };
    canvas.addEventListener('mousemove', pick);
    canvas.addEventListener('click', event => { pick(event); if (data) activateToken(token); });
    canvas.addEventListener('keydown', event => {
      if (!data || !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
      event.preventDefault();
      token += event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
      channel += event.key === 'ArrowDown' ? 1 : event.key === 'ArrowUp' ? -1 : 0;
      drawHeatmaps(); activateToken(token);
    });
  }
  el('kvFrame').addEventListener('change', () => { if (result) window.selectFrame(Number(el('kvFrame').value)); });
  ['kvLayer', 'kvHead', 'kvElevation', 'kvAzimuth', 'kvSharedScale'].forEach(id => el(id).addEventListener('change', refresh));
  return {reset, configure, selectFrame, highlightToken};
})();
