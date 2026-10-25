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
│   └── longva  vendored LongVA model dependency (installed separately)
├── model_zoo   optional local Video-LLM checkpoints
├── results     evaluation results
└── video_qa    code for StreamingVQA & OfflineVQA
```

ReKV modules are imported from `model.rekv` (for example,
`model.rekv.llava_onevision_rekv` and `model.rekv.patch`); SuPr-KV modules
are imported from `model.suprkv`. SuPr-KV reuses the decoder patch,
attention primitives, and QA adapter from `model.rekv`.

## Preparation

Our setup: Ubuntu 22.04, CUDA 12.6, 8x Nvidia H800 (80GB)

- Clone this repo: `git clone https://github.com/Becomebright/ReKV.git`
- Prepare the conda environment: `bash prepare.sh`
- Pretrained Video-LLMs are loaded by Hugging Face model ID and downloaded automatically on first use (then reused from the Hugging Face cache). No manual download to `model_zoo/` is required. Set `HF_HOME` to customize the cache location. Supported models:
  - [llava-onevision-qwen2-0.5b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
  - [llava-onevision-qwen2-7b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf)
  - [llava-onevision-qwen2-72b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-hf)
  - [LongVA-7B](https://huggingface.co/lmms-lab/LongVA-7B)
  - [Video-LLaVA-7B-hf](https://huggingface.co/LanguageBind/Video-LLaVA-7B-hf)
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
# Normally, set it to the number of GPUs on your machine.
# Yet, llava_ov_72b needs 4x 80GB GPUs. So set num_chunks to num_gpus//4.
num_chunks=8

# Supported model: llava_ov_0.5b llava_ov_7b llava_ov_72b video_llava_7b longva_7b
model=llava_ov_0.5b

# Supported dataset: qaego4d egoschema cgbench mlvu activitynet_qa rvs_ego rvs_movie
# MLVU has an extremely long video (~9hr). Remove it in the annotation file if your system doesn't have enough RAM.
dataset=qaego4d

python -m video_qa.run_eval \
    --num_chunks $num_chunks \
    --model ${model} \
    --dataset ${dataset} \
    --sample_fps 0.5 \
    --n_local 15000 \
    --retrieve_size 64
```

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

The optional suprKV OneVision backend uses one GPU-resident model for sequential encoding and QA, with no KV offload. It supports summary-scored, per-layer FP16/8/4/2-bit caches, token pruning, and frame eviction under a persistent-memory budget. See [usage, memory accounting, and tests](model/suprkv/README.md). Dataset evaluation uses `python -m video_qa.run_eval --method suprkv`; run `python -m video_qa.run_eval --help` for available options.

Run `bash run_suprkv.sh` with its configured sample limit, or evaluate all annotated RVS-Ego videos with `python -m video_qa.run_eval --method suprkv --model llava_ov_7b --dataset rvs_ego --num_chunks 1 --max_samples 0 --sample_fps 0.5 --retrieve_size 64 --budget_tokens 4096`. SuPr-KV shares ReKV's dataset, sampling, retrieval, sample-limit and worker arguments; `--budget_tokens` is a separate cache budget, not an alias for ReKV's `--n_local`. SuPr-KV supports the streaming `rvs_ego` and `rvs_movie` routes, with one model per GPU worker and the same six CSV columns as ReKV stream output; method configuration and cache statistics are saved separately in `.stats.jsonl`. `SuPrKVStreamVQA` reuses ReKV's `BaseVQA` data loading and CSV output. `eval_rvs_ego` and `eval_rvs_movie` in `run_eval` each implement worker launch, result merging and scoring, selecting ReKV or SuPr-KV via `--method`; use `--only_eval` to score existing results. For predictions only, run `python -m video_qa.suprkv_stream_vqa --anno_path data/rvs/ego/ego4d_oe.json --save_dir results/suprkv_stream`; this worker does not merge or score results.

Offline evaluation also supports `mlvu`, `qaego4d`, `egoschema`, `activitynet_qa`, and `cgbench`, for example `python -m video_qa.run_eval --method suprkv --dataset qaego4d --max_samples 5`. `SuPrKVOfflineVQA` encodes every sampled frame and finalizes the last segment before answering all questions for that video. It preserves ReKV's open-ended and multiple-choice CSV fields, including question types, and writes the same separate cache statistics as the stream worker. For offline predictions only, run `python -m video_qa.suprkv_offline_vqa --anno_path data/qaego4d/test_mc.json --save_dir results/suprkv_offline`. Offline QA reuses ReKV's open-question method and multiple-choice prompt construction: retrieval uses only the original question, multiple-choice generation includes the options and `Best option: (` prefix, and generation limits are 1024 tokens for open questions and 16 for choices. These per-question limits override the runtime's `--max_new_tokens` without changing stream defaults or video compression. Both methods use greedy generation and EOS stopping; decoding and nonempty answer parsing follow ReKV. SuPr-KV continues to mark empty multiple-choice answers incorrect.
