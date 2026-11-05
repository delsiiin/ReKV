#!/usr/bin/env bash
set -euo pipefail

# Override settings by appending CLI options, e.g. bash run_streamkv.sh --max_samples 5
python -m video_qa.run_eval \
    --method streamkv \
    --model llava_ov_7b \
    --dataset rvs_movie \
    --num_chunks 8 \
    --sample_fps 0.5 \
    --segment_mode semantic \
    --min_segment_size 4 \
    --max_segment_size 64 \
    --segment_theta 0.99 \
    --use_segment_summary true \
    --n_local 15000 \
    --compression_ratio 0.6 \
    --compress_temp 1 \
    --retrieve_size 64 \
    --retrieve_temp 0.5 \
    --retrieve_local true \
    --retrieve_local_size 8 \
    "$@"
