"""Exercise linked controls, heatmap cells, CSV, and stale asynchronous responses."""
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('gjs'), 'GJS JavaScript runtime unavailable')
class FrameQueryUITests(unittest.TestCase):
    def test_interaction_and_request_lifecycle(self):
        script = r'''
const GLib = imports.gi.GLib;
const read = path => new TextDecoder().decode(GLib.file_get_contents(path)[1]);
function assert(value, message) { if (!value) throw new Error(message); }
function loadFunctions(source, names) {
  const lines = source.split('\n');
  const offset = pos => lines.slice(0, pos.line-1).reduce((n, s) => n+s.length+1, 0)+pos.column;
  for (const fn of Reflect.parse(source).body) {
    if (fn.type === 'FunctionDeclaration' && (!names || names.includes(fn.id.name)))
      globalThis[fn.id.name] = new Function('return ('+source.slice(offset({...fn.loc.start, column:0}), offset(fn.loc.end))+')')();
  }
}
class Element {
  constructor(tag = 'div', text = '') {
    this.tag = tag; this.textContent = text; this.children = []; this.dataset = {}; this.style = {};
    this.attrs = {}; this.handlers = {}; this._value = ''; this.disabled = false;
    this.classes = new Set(); this.classList = {add: x => this.classes.add(x), contains: x => this.classes.has(x)};
  }
  set value(v) { this._value = String(v); } get value() { return this._value; }
  get options() { return this.children; }
  append(...children) { this.children.push(...children); if (this.tag === 'select' && !this.value && this.children.length) this.value = this.children[0].value; }
  replaceChildren(...children) { this.children = []; if (this.tag === 'select') this.value = ''; this.append(...children); }
  addEventListener(event, fn) { (this.handlers[event] ??= []).push(fn); }
  emit(event, e = {}) { for (const fn of this.handlers[event] ?? []) fn(e); }
  setAttribute(k, v) { this.attrs[k] = v; if (k === 'data-token-index') this.dataset.tokenIndex = String(v); }
  querySelectorAll(selector) {
    const result = [];
    function walk(el, overlay = false) {
      overlay ||= el.classes?.has('attention-overlay');
      if (overlay && el.tag === 'rect') result.push(el);
      for (const child of el.children ?? []) walk(child, overlay);
    }
    walk(this); return result;
  }
}
const elements = {};
globalThis.$ = id => elements[id] ??= new Element(/(Segment|Frame|Layer|Summary|Rope)$/.test(id) ? 'select' : 'div');
globalThis.node = (tag, cls = '', text = '') => { const el = new Element(tag, text); if (cls) el.classes.add(cls); return el; };
globalThis.svgNode = (tag, attrs = {}, text = '') => { const el = node(tag, '', text); for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v); return el; };
globalThis.fmt = value => String(value);
globalThis.document = new Element();
globalThis.AbortController = class { constructor() { this.signal = {aborted: false}; } abort() { this.signal.aborted = true; } };
globalThis.URLSearchParams = class { constructor(values) { this.values = values; } toString() { return Object.entries(this.values).map(([k,v]) => `${k}=${v}`).join('&'); } };
globalThis.frameQueryState = {controller: null, request: 0, selected: 0, data: null};
globalThis.segmentAttentionState = {controller: null, request: 0, selected: 0, data: null};
globalThis.attentionMeasures = [{key: 'pre_softmax', label: 'Softmax 前'}, {key: 'post_softmax', label: 'Softmax 后'}];
globalThis.ropeLabel = () => 'RoPE 前';
globalThis.state = {result: {
  model: {layers: 2}, frames: [0,1,2].map(index => ({index, timestamp: index, width: 10, height: 10, image_url: '/frame.jpg'})),
  spatial: {cols: 2, cells: [{row:0, col:0, box:[0,0,.5,1]}, {row:0, col:1, box:[.5,0,1,1]}]},
  segment_experiment: {frame_query_attention: {url:'/api/frame-query-attention/abc'}, segment_attention: {url:'/summary'},
    segments: [{id:0, frame_indices:[0,1]}, {id:1, frame_indices:[2]}]}
}};
const app = read('observation/dist/app.js'), attention = read('observation/dist/attention.js'), frame = read('observation/dist/frame-query.js');
loadFunctions(app, ['frameMeanPruning', 'similarityColorRange', 'cosineColor']);
loadFunctions(attention, ['attentionFrames', 'renderAttentionMap']); loadFunctions(frame);
new Function(frame.slice(frame.indexOf("$('exportFrameQueryCsv').addEventListener")))();
let summaryRenders = 0, frameRenders = 0;
const render = renderFrameQuery;
globalThis.renderSegmentAttention = () => summaryRenders++;
globalThis.renderFrameQuery = () => frameRenders++;
initFrameQuery();
assert(!$('frameQueryPanel').hidden && $('frameQueryLayer').value === 'all', 'initialize default layer');
$('segmentAttentionSegment').value = '0'; $('segmentAttentionFrame').value = '0'; $('segmentAttentionLayer').value = 'all';
$('segmentAttentionSummary').value = '1'; $('segmentAttentionRope').value = 'post';
$('frameQuerySegment').value = '1'; $('frameQuerySegment').emit('change');
assert($('segmentAttentionSegment').value === '1' && $('segmentAttentionFrame').value === '2', 'frame to summary segment/frame');
assert($('segmentAttentionSummary').value === '1' && $('segmentAttentionRope').value === 'post', 'preserve summary/rope choices');
$('segmentAttentionSegment').value = '0'; attentionFrames(); $('segmentAttentionFrame').value = '1'; $('segmentAttentionLayer').value = '1';
$('segmentAttentionLayer').emit('change');
assert($('frameQuerySegment').value === '0' && $('frameQueryFrame').value === '1' && $('frameQueryLayer').value === '1', 'summary to frame controls');
assert(summaryRenders === 1 && frameRenders === 3, 'no event recursion');
globalThis.renderFrameQuery = render;
const data = {segment_index:0, frame_index:1, layer:'1', rope:'pre', query_count:2, image_tokens:2, segment_tokens:4,
  frame_token_start:4, pre_softmax:[-2,2], post_softmax:[.1,.3], frame_masses:[{frame_index:1,mass:.4}], segment_mass:1};
const pending = [];
globalThis.fetch = (url, options) => new Promise(resolve => pending.push({url, options, resolve}));
const response = value => ({ok:true, json:async () => value});
async function main() {
  const first = render(); const second = render();
  assert(pending[0].options.signal.aborted, 'cancel prior request');
  assert(pending[1].url.includes('segment=0&frame=1&layer=1'), 'API selection');
  pending[1].resolve(response(data)); await second;
  pending[0].resolve(response({...data, query_count:99})); await first;
  assert(frameQueryState.data.query_count === 2, 'ignore late older response');
  assert(!$('exportFrameQueryCsv').disabled && $('frameQueryMaps').children.length === 2, 'both measures render');
  const rects = $('frameQueryMaps').querySelectorAll('.attention-overlay rect');
  assert(rects.length === 4, 'two frame grids');
  rects[1].emit('click');
  assert(frameQueryState.selected === 1 && $('frameQueryReadout').textContent.includes('token #1'), 'token click readout');
  assert(rects[1].attrs.stroke === '#ff8a24' && rects[3].attrs.stroke === '#ff8a24', 'linked token highlight');
  const csv = frameQueryCsv(data).split('\n').map(row => row.split(','));
  assert(csv.length === 3 && csv[1][9] === '-2' && csv[1][11] === '0' && csv[1][13] === '1', 'CSV logits threshold/pruning');
  assert(csv[2][10] === '0.3' && csv[2][13] === '0', 'CSV probability and retained token');
  const third = render(); clearFrameQuery(); pending[2].resolve(response(data)); await third;
  assert(frameQueryState.data === null && $('frameQueryPanel').hidden, 'clear prevents late response from repopulating panel');
  delete state.result.segment_experiment.frame_query_attention;
  initFrameQuery(); assert($('frameQueryMeta').textContent.includes('重新生成') && $('frameQueryFrame').disabled, 'legacy cache message');
  state.result = {}; initFrameQuery(); assert($('frameQueryPanel').hidden, 'no segment mode hides panel');
  print('frame-query UI OK');
}
const loop = GLib.MainLoop.new(null, false);
let failure = null;
main().then(() => loop.quit(), error => {failure=error; loop.quit();}); loop.run();
if (failure) throw failure;
'''
        result = subprocess.run(['gjs', '-c', script], cwd=Path(__file__).resolve().parents[1],
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
