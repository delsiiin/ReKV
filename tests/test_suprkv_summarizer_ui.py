"""Exercise observation controls and legacy-result rendering without a browser."""
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('gjs'), 'GJS JavaScript runtime unavailable')
class SummarizerUITests(unittest.TestCase):
    def test_controls_payload_and_old_result_compatibility(self):
        script = r'''
const GLib = imports.gi.GLib;
function read(path) { return new TextDecoder().decode(GLib.file_get_contents(path)[1]); }
const app = read('observation/dist/app.js'), results = read('observation/dist/suprkv.js');
Reflect.parse(app); Reflect.parse(results);
function load(source, name) {
  const fn = Reflect.parse(source).body.find(n => n.type === 'FunctionDeclaration' && n.id.name === name);
  const lines = source.split('\n');
  const start = lines.slice(0, fn.loc.start.line - 1).join('\n').length + (fn.loc.start.line > 1 ? 1 : 0) + source.split('\n')[fn.loc.start.line - 1].indexOf('function ');
  const end = lines.slice(0, fn.loc.end.line - 1).join('\n').length + (fn.loc.end.line > 1 ? 1 : 0) + fn.loc.end.column;
  return new Function('return (' + source.slice(start, end) + ')')();
}
function assert(value, message) { if (!value) throw new Error(message); }
const elements = {};
globalThis.$ = id => elements[id] ??= {value: '2', checked: true, options: [{}],
  classList: {remove() {}}, querySelectorAll() { return [{value: '0'}]; },
  replaceChildren(...children) { this.children = children; }, append() {}};
globalThis.state = {result: null};
globalThis.document = {querySelector() { return {}; }};
$('mode').value = 'compression'; $('compressionSummarizer').value = 'main';
$('compressionCustomQuestions').value = ''; $('maxFrames').value = '';
const parameters = load(app, 'parameters'), update = load(app, 'updateWindowLabels');
assert(parameters().summarizer_model_path === null, 'main default');
$('compressionSummarizer').value = 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf';
assert(parameters().summarizer_model_path.endsWith('0.5b-ov-hf'), 'preset model ID');
$('compressionSummarizer').value = 'custom'; $('compressionSummarizerPath').value = '  /models/custom  ';
update();
assert(parameters().summarizer_model_path === '/models/custom', 'trim custom path');
assert($('compressionSummarizerPath').required && !$('compressionSummarizerPath').disabled, 'custom path required');
$('compressionScoring').value = 'frame_mean_query'; update();
assert(parameters().scoring_strategy === 'frame_mean_query', 'strategy in payload');
assert(parameters().summarizer_model_path === null, 'unused summarizer excluded');
assert($('compressionSummarizer').disabled && !$('compressionSummarizerPath').required, 'skip summary controls');
$('compressionScoring').value = 'summary'; update();
assert(!$('compressionSummarizer').disabled && $('compressionSummarizerPath').required, 'restore summary controls');
$('mode').value = 'video'; update();
assert(!('summarizer_model_path' in parameters()), 'ordinary observation excludes summarizer');
assert($('compressionSummarizerPath').disabled && !$('compressionSummarizerPath').required, 'hidden field must not block submit');

for (const name of ['stopCachePlayback', 'prepareCompressionQuestions', 'renderCompressionAnswer',
  'renderRetrievalHits', 'prepareCompressionFrameStates', 'renderCompressionCostChart',
  'renderCompressionMemoryChart', 'renderCompressionFrameStates']) globalThis[name] = () => {};
globalThis.cachePlayback = {rowSignatures: new Map()};
for (const name of ['fmt', 'mib', 'pct', 'millis', 'deltaMs']) globalThis[name] = x => String(x);
globalThis.metricCard = () => ({});
globalThis.compressionTable = (headers, rows) => ({headers, rows});
const render = load(results, 'renderCompressionResult');
const run = () => ({status: 'complete', cache: {counts: {compressions: 0, evicted_frames: 0}, budget_bytes: 100, retained_frames: 2}});
const e = {frame_count: 2, method: {cache_device: 'cpu'}, config: {}, runs: {before: run(), after: run()}, memory: {}, stages: []};
const result = {compression_experiment: e, frames: []};
render(result, '/result.json', true);
assert($('compressionLabel').textContent.includes('1 → 1 个模型'), 'legacy model count');
assert($('compressionLabel').textContent.includes('摘要配置：主模型'), 'legacy default summarizer');
e.config.summarizer_model_path = 'auxiliary';
e.runs.before.model_stats = {model_instances: 1, model_parameter_bytes: 100};
e.runs.after.model_stats = {model_instances: 2, model_parameter_bytes: 150, summary_retries: 3, summary_failures: 1};
render(result, '/result.json', false);
assert($('compressionLabel').textContent.includes('1 → 2 个模型'), 'independent model count');
assert($('compressionLabel').textContent.includes('重试 3 次 / 最终失败 1 次'), 'fallback counters');
e.runs.after.commits = [{segment_id: 0, summary: 'real caption', summary_source: 'main-checkpoint',
  summarizer_error: 'empty auxiliary summary', summary_retried: true, summary_retry_succeeded: true, summary_error: ''}];
render(result, '/result.json', false);
assert(!$('compressionSummariesPanel').hidden, 'show segment summaries');
assert($('compressionSummariesTable').children[0].rows[0][1] === 'main-checkpoint', 'show actual fallback source');
assert($('compressionSummariesTable').children[0].rows[0][4] === '成功', 'show retry outcome');
e.config.scoring_strategy = 'frame_mean_query';
e.stages = [{label: '生成 summary', before_seconds: 0, after_seconds: 0, delta_seconds: 0, enabled: false}];
render(result, '/result.json', false);
assert($('compressionLabel').textContent.includes('摘要配置：未启用'), 'summary disabled label');
assert($('compressionCostTable').children[0].rows[0][2] === '未启用', 'summary stage disabled');
print('UI behavior OK');
'''
        result = subprocess.run(['gjs', '-c', script], cwd=Path(__file__).resolve().parents[1],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
