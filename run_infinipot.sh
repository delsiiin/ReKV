#!/usr/bin/env bash
set -euo pipefail

gpus="${CUDA_VISIBLE_DEVICES:-0}"
# InfiniPot-V/scripts/run_sample.sh experiment profile; append CLI overrides.
python -m video_qa.run_eval \
    --gpus "$gpus" \
    --method infinipot \
    --model qwen2_5_vl_7b \
    --dataset mlvu \
    --num_chunks 1 \
    --sample_fps 2 \
    --block_size 32 \
    --compress_frame_num 24 \
    --compression_method infinipot-v \
    --max_frames_num 768 \
    --tar_ratio 0.5 \
    --query_ratio 0.25 \
    "$@"
