#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# One worker/model per visible GPU; 0 evaluates all annotated videos.
num_chunks=1
max_samples=0
model=llava_ov_7b
dataset=rvs_ego

python -m video_qa.run_eval \
    --method suprkv \
    --num_chunks "$num_chunks" \
    --max_samples "$max_samples" \
    --model "$model" \
    --dataset "$dataset" \
    --sample_fps 0.5 \
    --retrieve_size 64 \
    --budget_tokens 8192 \
    --max_segment_frames 32 \
    --recent_frames 2 \
    --min_segment_frames 4 \
    "$@"