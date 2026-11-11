# ReKV

Official PyTorch code of "Streaming Video Question-Answering with In-context Video KV-Cache Retrieval", *ICLR* 2025.

## Abstract

We propose **ReKV**, a novel, training-free approach that integrates seamlessly with existing Video Large Language Models (Video-LLMs) to enable efficient streaming video question-answering (**StreamingVQA**).

Traditional VideoQA systems struggle with long videos, as they must process the entire video before responding to queries, and repeat this process for each new question. In contrast, our approach analyzes long videos in a streaming fashion, allowing for prompt responses as soon as user queries are received. 
- Building on a common Video-LLM, we first incorporate a sliding-window attention mechanism, ensuring that input frames attend to a limited number of preceding frames, thereby reducing computational overhead.
- To prevent information loss, we store processed video key-value caches (KV-Caches) in RAM and disk, reloading them into GPU memory as needed. 
- Additionally, we introduce a retrieval method that leverages an external retriever or the parameters within Video-LLMs to retrieve only query-relevant KV-Caches, ensuring both efficiency and accuracy in question answering.

ReKV enables the separation of video analyzing and question-answering across different processes and GPUs, significantly enhancing the efficiency of StreamingVQA. 
Through comprehensive experimentation, we validate the efficacy and practicality of our approach, which significantly boosts efficiency and enhances applicability over existing VideoQA models.

## Directory Structure

```
.
├── data        processed benchmarks
├── model
│   ├── rekv    ReKV model adapters, attention, and retrieval cache
│   ├── suprkv  SuPr-KV runtime, compression, and retrieval
│   ├── streamkv semantic segments, guided compression, adaptive retrieval
│   ├── infinipot Qwen block prefill and TaR + Value Norm KV compression
│   └── native original model inference
├── model_zoo   optional local Video-LLM checkpoints
├── results     evaluation results
└── video_qa    code for StreamingVQA & OfflineVQA
```

ReKV modules are imported from `model.rekv` (for example,
`model.rekv.llava_onevision_rekv` and `model.rekv.patch`); SuPr-KV modules
are imported from `model.suprkv`. SuPr-KV reuses the decoder patch,
attention primitives, and QA adapter from `model.rekv`.

`video_qa/model_paths.py` contains the single model registry, `MODEL_PATHS`.
Every listed model supports `native`, `rekv`, and `suprkv` on
both offline and streaming routes. Architecture resolution is shared across
methods; there are no separate method-specific model lists.

`streamkv` supports the same seven datasets as the existing methods, using
the LLaVA-OneVision presets on both routes. Its segmentation, budget
allocation, attention, generation and CUDA cache manager follow the supplied
`StreamKV/` source. Streaming questions sharing the same sampled prefix reuse
the video archive and retrieve independently; a changed prefix is re-encoded.
This retains one archive without cloning a second full video KV snapshot.
See [StreamKV source adaptation](model/streamkv/README.md) for the Python CLI
defaults, the separate reference experiment profile, and differential tests.

```bash
bash run_streamkv.sh --num_chunks 1 --max_samples 5
python -m video_qa.run_eval --method streamkv --model llava_ov_7b --dataset mlvu --max_samples 5
```

`infinipot` adapts the supplied InfiniPot-V implementation to all seven dataset
routes using Qwen2.5-VL and LLaVA-OneVision presets. The Python CLI defaults use
32 temporal units per block and retain 24: Qwen units are temporal grids,
whereas OneVision units are sampled frames (normally 196 tokens each). Both
sample up to 768 raw frames at 2 FPS and generate up to 300 tokens. RVS questions
use only their visible video prefix. Both models reuse video prefill across
questions sharing that prefix, in block and native video generation paths, and
clone KV for each answer. One extra video KV snapshot remains on the model
device; a new prefix rebuilds it. Native decoding retains its generation settings.
The shell script supplies its own budget
settings; append explicit overrides when comparing models.
See [algorithm flow, source parameters, and validation](model/infinipot/README.md).

```bash
bash run_infinipot.sh --dataset mlvu --max_samples 5
bash run_infinipot.sh --dataset rvs_movie --max_samples 1
# OneVision: 41 * 196 = 8036 visual KV tokens, within an 8192-token budget.
bash run_infinipot.sh --model llava_ov_7b --block_size 41 --compress_frame_num 31
```

## Preparation

Our setup: Ubuntu 22.04, CUDA 12.6, 8x Nvidia H800 (80GB)

- Clone this repo: `git clone https://github.com/Becomebright/ReKV.git`
- Prepare the conda environment: `bash prepare.sh`
- Pretrained Video-LLMs are loaded by Hugging Face model ID and downloaded automatically on first use (then reused from the Hugging Face cache). No manual download to `model_zoo/` is required. Set `HF_HOME` to customize the cache location. Supported models:
  - [llava-onevision-qwen2-0.5b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
  - [llava-onevision-qwen2-7b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf)
  - [llava-onevision-qwen2-72b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-hf)
  - [Video-LLaVA-7B-hf](https://huggingface.co/LanguageBind/Video-LLaVA-7B-hf)
  - Qwen2.5-VL Instruct: `qwen2_5_vl_3b`, `qwen2_5_vl_7b`, `qwen2_5_vl_72b` (all methods: native, ReKV and SuPr-KV)
- Download benchmarks under `data/`
  - [MLVU-dev-mc](https://huggingface.co/datasets/MLVU/MVLU)
  - [QAEgo4D-test-mc](https://huggingface.co/datasets/Becomebright/QAEgo4D-MC-test/tree/main)
  - [EgoSchema-full](https://huggingface.co/datasets/lmms-lab/egoschema)
  - [ActivityNet-QA](https://huggingface.co/datasets/lmms-lab/ActivityNetQA)
  - [RVS](https://huggingface.co/datasets/Becomebright/RVS)
  - [CGBench](https://huggingface.co/datasets/CG-Bench/CG-Bench)
  - The `data/` folder should be arranged as:
    ```
    data
    ├── activitynet_qa
    │   ├── test.json
    │   └── videos
    ├── cgbench
    │   ├── full_mc.json
    │   └── videos
    ├── egoschema
    │   ├── full.json
    │   └── videos
    ├── mlvu
    │   ├── dev_debug_mc.json
    │   └── videos
    ├── qaego4d
    │   ├── test_mc.json
    │   └── videos
    └── rvs
        ├── ego
        │   ├── ego4d_oe.json
        │   └── videos
        └── movie
            ├── movienet_oe.json
            └── videos
    ```
- Increases the memory map limit for processes (needed for offloading KV-Caches): `sudo sysctl -w vm.max_map_count=262144`

## Evaluation

ReKV and SuPr-KV share the original ReKV sampling rules in
`video_qa/sampling.py`: regular videos use a fixed stride of
`int(round(source_fps) / sample_fps)`, and 1-FPS NPY videos use
`np.linspace(0, total - 1, int(total * sample_fps), dtype=int)`.
These frame selections also match Observation before its start-time/frame-count
filters. RVS questions use the original ReKV array boundary:
`video[:int(end_time * sample_fps)]`. Only newly visible sampled frames are
encoded between questions. Both methods read and encode one sampled frame at a
time, then answer using the resulting cache without a second timestamp
filter. The boundary is an array index, not a guarantee that source timestamps
are strictly less than the annotated `end_time`.

ReKV offline evaluation also reads and encodes one frame at a time, completing
the video before answering its questions. Neither ReKV evaluation entry point
materializes the whole video or question interval as a raw-frame array. This
aligns input buffering with SuPr-KV; ReKV still retains its full historical CPU
KV cache with the same `n_local` and retrieval settings. No SuPr-KV quantization,
pruning, eviction, or GPU cache budget is applied to ReKV. Historical KV RAM
therefore continues to grow with video length. Single-frame forwards can differ
slightly from batched forwards due to floating-point arithmetic.

For a local HTML experiment showing RVS sampled frames, last-32-token attention
importance and frame-conditional entropy, run `bash run_observation.sh` in the
`rekv` environment and open `http://127.0.0.1:8765`.
See [the observation guide](observation/README.md) for inputs and metric definitions.

```bash
# The number of processes utilized for parallel evaluation.
# One GPU per process for every method and model size.
num_chunks=8
gpus=0,1,2,3,4,5,6,7

# Supported model: llava_ov_0.5b llava_ov_7b llava_ov_72b video_llava_7b qwen2_5_vl_3b qwen2_5_vl_7b qwen2_5_vl_72b
model=llava_ov_0.5b

# Supported dataset: qaego4d egoschema cgbench mlvu activitynet_qa rvs_ego rvs_movie
# MLVU has an extremely long video (~9hr). Remove it in the annotation file if your system doesn't have enough RAM.
dataset=qaego4d

python -m video_qa.run_eval \
    --gpus "$gpus" \
    --num_chunks $num_chunks \
    --model ${model} \
    --dataset ${dataset} \
    --sample_fps 0.5 \
    --n_local 15000 \
    --retrieve_size 64
```

Every eval worker loads its complete model on its local `cuda:0` using the shared
single-device loader. `--num_chunks` controls the process count (capped by the
number of selected videos); `--gpus` lists GPU IDs in worker order. For example:

```bash
# Worker 0 uses GPU 2; worker 1 uses GPU 5.
bash run_rekv.sh --num_chunks 2 --gpus 2,5
CUDA_VISIBLE_DEVICES=2,5 bash run_suprkv.sh --num_chunks 2
```

The launcher uses `--gpus` first, then `CUDA_VISIBLE_DEVICES`, then `0,1,...` when
neither is set. Extra GPU entries are unused; too few entries fail before workers
start. GPU UUIDs and MIG identifiers are also accepted. Eval uses local `cuda:0`;
select physical GPUs with the GPU list, not `--device cuda:N`. All model sizes,
including 72B, must fit on one GPU per worker; there is no automatic model sharding
or weight offload. Model-specific precision and attention implementations remain
unchanged. The scripts expose a `gpus` variable and accept CLI overrides.

## Citation

```latex
@inproceedings{di2025rekv,
  title={Streaming Video Question-Answering with In-context Video KV-Cache Retrieval},
  author={Di, Shangzhe and Yu, Zhelun and Zhang, Guanghao and Li, Haoyuan and Cheng, Hao and Li, Bolin and He, Wanggui and Shu, Fangxun and Jiang, Hao and others},
  booktitle={ICLR},
  year={2025}
}
```

## Acknowledgements

Our code is based on [InfLLM](https://github.com/thunlp/InfLLM), [StreamingLLM](https://github.com/mit-han-lab/streaming-llm), and [Flash-VStream](https://github.com/IVGSZ/Flash-VStream).

## suprKV: Summary-guided progressive KV compression

The optional suprKV backend supports every model in `MODEL_PATHS`: OneVision, Video-LLaVA and Qwen2.5-VL. It uses one GPU-resident model for sequential encoding and QA, with no KV offload. It supports summary-scored, per-layer FP16/8/4/2-bit caches, token pruning, and frame eviction under a persistent-memory budget. See [usage, memory accounting, and tests](model/suprkv/README.md). Dataset evaluation uses `python -m video_qa.run_eval --method suprkv`; run `python -m video_qa.run_eval --help` for available options.

Run `bash run_suprkv.sh` with its configured sample limit, or evaluate all annotated RVS-Ego videos with `python -m video_qa.run_eval --method suprkv --model llava_ov_7b --dataset rvs_ego --num_chunks 1 --max_samples 0 --sample_fps 0.5 --retrieve_size 64 --budget_tokens 4096`. SuPr-KV shares ReKV's dataset, sampling, retrieval, sample-limit and worker arguments; `--budget_tokens` is a separate cache budget, not an alias for ReKV's `--n_local`. SuPr-KV supports the streaming `rvs_ego` and `rvs_movie` routes, with one main model per GPU worker and the same six CSV columns as ReKV stream output; method configuration and cache statistics are saved separately in `.stats.jsonl`. `SuPrKVStreamVQA` uses the shared `BaseVQA` and streaming traversal. All dataset routes share worker launch, result merging and scoring in `run_eval`, selecting an inference method via `--method`; use `--only_eval` to score existing results. For predictions only, run `python -m video_qa.suprkv_stream_vqa --anno_path data/rvs/ego/ego4d_oe.json --save_dir results/suprkv_stream`; this worker does not merge or score results.

To generate segment summaries with a separate small OneVision model, add
`--summarizer-model-path llava-hf/llava-onevision-qwen2-0.5b-ov-hf` (or a local model directory).
The main model still extracts scoring queries from the resulting text. The auxiliary model shares its device/dtype,
is loaded once per worker, and falls back to the main model once on recoverable summary failure.
Observation's **KV 压缩前后对照 → 摘要模型** offers the same option; ordinary observation is unchanged.


Offline evaluation also supports `mlvu`, `qaego4d`, `egoschema`, `activitynet_qa`, and `cgbench`, for example `python -m video_qa.run_eval --method suprkv --dataset qaego4d --max_samples 5`. `SuPrKVOfflineVQA` encodes every sampled frame and finalizes the last segment before answering all questions for that video. It preserves ReKV's open-ended and multiple-choice CSV fields, including question types, and writes the same separate cache statistics as the stream worker. For offline predictions only, run `python -m video_qa.suprkv_offline_vqa --anno_path data/qaego4d/test_mc.json --save_dir results/suprkv_offline`. Offline QA uses the shared open-question method and multiple-choice prompt construction: retrieval uses only the original question, multiple-choice generation includes the options and `Best option: (` prefix, and generation limits are 1024 tokens for open questions and 16 for choices. These per-question limits override the runtime's `--max_new_tokens` without changing stream defaults or video compression. Both methods use greedy generation and EOS stopping; decoding and nonempty answer parsing follow ReKV. SuPr-KV continues to mark empty multiple-choice answers incorrect.


## Qwen2.5-VL with all inference methods

All three methods (`native`, `rekv`, `suprkv`), on both offline and streaming
routes, accept `qwen2_5_vl_3b`, `qwen2_5_vl_7b`,
and `qwen2_5_vl_72b`. The presets load the corresponding
`Qwen/Qwen2.5-VL-{3B,7B,72B}-Instruct` checkpoint. Use `--model_path /path/to/checkpoint`
to load local weights while retaining the matching `--model` preset.

```bash
python -m video_qa.run_eval --method native --model qwen2_5_vl_7b \
    --dataset rvs_ego --max_samples 5
python -m video_qa.run_eval --method rekv --model qwen2_5_vl_7b \
    --dataset rvs_ego --max_samples 5 --n_local 15000
python -m video_qa.run_eval --method suprkv --model qwen2_5_vl_7b \
    --dataset rvs_ego --max_samples 5 --budget_tokens 8192
```

The ReKV/SuPr-KV adapters use fixed 392×392 RGB frames and 196 visual tokens per sampled frame.
Each frame is independently padded to one temporal patch by the Qwen video processor;
adjacent sampled frames are never merged. This keeps retrieval indices, recent-frame
protection and memory accounting aligned with the existing frame-based algorithms.
ReKV requires `n_local >= 196` and automatically bounds encoding chunks by this window.

**Position encoding:** retrieval encoding and QA use the existing ReKV/SuPr-KV
one-dimensional RoPE relocation, with Qwen's checkpoint frequencies. They do not
reproduce native full-video temporal/spatial MRoPE. SuPr-KV's isolated main-model
summaries use native spatial MRoPE with each sampled frame represented as an image,
and continue decoding at the corresponding RoPE offset. Both scoring strategies
(`summary`, `frame_mean_query`) and compression granularities (`layer`, `head`) are supported.
An optional independent summary model still uses the OneVision loader.

**Native Qwen2.5-VL:** `--method native` loads the original
`Qwen2_5_VLForConditionalGeneration` and calls its processor and `generate()`.
The checkpoint's dynamic spatial resizing, temporal patch merging, MRoPE, and
generation cache are preserved. It retains sampled RGB frames on CPU and processes
the complete visible video anew for each question, including native padding for
odd frame counts. Sampling is performed once by the shared benchmark sampler;
the processor receives the resulting FPS inferred from source timestamps (the
requested `--sample_fps` for a single frame). Question caches and RoPE deltas are
cleared between answers. Native visual token counts may differ from the fixed
196-token-per-frame retrieval adapters.

For direct Python use, set both fields:
`StreamingConfig(model_path='Qwen/Qwen2.5-VL-7B-Instruct', model_family='qwen2_5_vl')`.
All methods keep the model on one selected device; the 72B preset therefore requires
enough memory on that device. Eval assigns one GPU per worker via `--gpus` or
`CUDA_VISIBLE_DEVICES`. Native and SuPr-KV direct worker entry points additionally
accept `--device` for local single-device checks.

CPU regression tests use tiny randomly initialized models without downloads:
`python -m unittest discover -s tests -p 'test_qwen2_5_vl*.py' -v`.
The real ReKV ContextManager smoke test additionally requires CUDA.

## Original language-model inference and method extensions

Use `--method native` for full-context inference with the original language model.
For OneVision and Video-LLaVA, this is a baseline for the **LLM processing stage**: frame sampling, vision encoder,
visual token pooling and prompts remain the same as the existing model adapters.
The decoder is loaded from the original model class with no ReKV/SuPr-KV attention
patch, retrieval, quantization, token pruning or cache eviction. Qwen2.5-VL uses its
original multimodal processor and full generation model as described above.
All presets in `video_qa/model_paths.py` support native inference.

```bash
# Offline benchmark, first five videos
python -m video_qa.run_eval --method native --model llava_ov_7b \
    --dataset qaego4d --sample_fps 0.5 --max_samples 5

# Streaming benchmark: only frames visible at each question's cutoff
python -m video_qa.run_eval --method native --model llava_ov_7b \
    --dataset rvs_ego --sample_fps 0.5 --max_samples 5

# Predictions only, without merging or invoking a scorer
python -m video_qa.native_offline_vqa --model llava_ov_7b \
    --anno_path data/qaego4d/test_mc.json --save_dir results/native_test
```

For OneVision and Video-LLaVA, visual features are encoded once per sampled
frame; Qwen2.5-VL reprocesses the visible sampled video per question. Each question prefills the
complete visible video prefix and its question from scratch, then generates greedily
with the model's native KV cache and EOS stopping. Question/answer tokens are never
carried into another question. Streaming uses the same `int(end_time * sample_fps)`
sampled-array boundary as the other methods, including repeated and zero cutoffs.
Full context increases memory use with video length; this path does not silently
truncate frames or switch to a compressed method.

Native results go to `results/native/<model>/<dataset>/fps_<fps>` (with the usual
`/samples_<N>` suffix). They contain the shared QA fields without retrieval columns.
`--model_path` can select local weights; `--device cpu` is available in direct worker
entry points for small-model checks. Native `--max_new_tokens` overrides all answer limits; otherwise the defaults
are 1024 for offline open answers, 256 for streaming open answers and 16 for choices.
Retrieval and compression options such as `--retrieve_size`, `--n_local` and
`--budget_tokens` are rejected for native inference. Existing ReKV and SuPr-KV
commands, result directories and CSV metadata remain supported.

The extension points are:

- `video_qa/base.py`: common prompts, answer records, CSV output and worker setup.
- `video_qa/flows.py`: shared offline/streaming traversal and frame boundaries.
- `video_qa/methods/`: method-owned CLI options, validation, lazy model loading,
  resource cleanup and result layout. GPU assignment is shared in `run_eval` and
  single-device model loading in `model/loading.py`. Register a method in
  `METHODS`; the launcher and base worker need no new method branches.
- Method entry points combine an offline/streaming flow with lifecycle hooks:
  `start_video`, `consume_frame`, optional `finish_encoding` and `finish_video`.
  A runtime supplies `get_prompt` and `question_answering`; methods can override
  the QA hook when their streaming API has different semantics.
- `model/video_backbone.py`: shared model-family prompts and visual feature extraction.
  `model/native/runtime.py` owns full-context LLM prefill and generation.

A new method implements the `Method` interface in `video_qa/methods/common.py`,
registers its instance, and supplies `<method>_offline_vqa` and `<method>_stream_vqa`
entry points (or overrides `solver`). Its `analyzer_kwargs` can add method-specific
record metadata without introducing retrieval assumptions into `BaseVQA`.
