#!/usr/bin/env bash
set -euo pipefail

# 默认执行顺序：可删除不需要的方法，或调整顺序。
default_methods=(rekv suprkv streamkv infinipot)

# 各方法独立配置。可分别修改 GPU、模型、数据集及方法专属参数。
# --max_samples 0 表示全量评估；每个 worker 使用一张 GPU。
# GPU 默认继承 CUDA_VISIBLE_DEVICES，否则使用 GPU 1。
rekv_args=(
    --gpus "${CUDA_VISIBLE_DEVICES:-1}"
    --num_chunks 1
    --max_samples 0
    --model llava_ov_7b
    --dataset rvs_ego
    --sample_fps 0.5
    --n_local 15000
    --retrieve_size 64
)

suprkv_args=(
    --gpus "${CUDA_VISIBLE_DEVICES:-1}"
    --num_chunks 1
    --max_samples 0
    --model llava_ov_7b
    --dataset rvs_movie
    --sample_fps 0.5
    --retrieve_size 64
    --budget_tokens 8192
    --max_segment_frames 32
    --recent_frames 2
    --min_segment_frames 4
    --scoring_strategy frame_mean_query
    --scoring_scope frame
    --frame-score-batch-size 8
    --summary_max_new_tokens 128
    --max_new_tokens 256
    --compression-granularity head
)

streamkv_args=(
    --gpus "${CUDA_VISIBLE_DEVICES:-1}"
    --model llava_ov_7b
    --dataset rvs_ego
    --num_chunks 1
    --sample_fps 0.5
    --segment_mode semantic
    --min_segment_size 4
    --max_segment_size 64
    --segment_theta 0.99
    --use_segment_summary true
    --n_local 15000
    --compression_ratio 0.6
    --compress_temp 1
    --retrieve_size 64
    --retrieve_temp 0.5
    --retrieve_local true
    --retrieve_local_size 8
    --max_samples 0
)

infinipot_args=(
    --gpus "${CUDA_VISIBLE_DEVICES:-1}"
    --model llava_ov_7b
    --dataset rvs_ego
    --num_chunks 1
    --sample_fps 2
    --block_size 41
    --compress_frame_num 31
    --compression_method infinipot-v
    --max_frames_num 768
    --tar_ratio 0.5
    --query_ratio 0.25
    --max_samples 0
)

usage() {
    cat <<'EOF'
用法: bash run_methods.sh [--dry-run] [方法 ...] [-- 评估参数 ...]

方法: rekv suprkv streamkv infinipot
不指定方法时，按照脚本顶部 default_methods 的顺序执行。
各方法的参数在对应的 *_args 数组中独立设置。
-- 后的参数覆盖本次所有选中方法的配置。
--dry-run 只打印命令，不启动评估。任一方法失败时停止执行。

示例:
  bash run_methods.sh
  bash run_methods.sh suprkv rekv
  bash run_methods.sh --dry-run
  bash run_methods.sh rekv streamkv -- --max_samples 5 --gpus 0
EOF
}

methods=()
dry_run=false
while (($#)); do
    case "$1" in
        --dry-run) dry_run=true ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        rekv|suprkv|streamkv|infinipot) methods+=("$1") ;;
        *) printf '未知方法或选项: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
if ((${#methods[@]} == 0)); then
    methods=("${default_methods[@]}")
fi

# 在启动任何评估之前检查完整方法列表。
for method in "${methods[@]}"; do
    case "$method" in
        rekv|suprkv|streamkv|infinipot) ;;
        *) printf '未知方法: %s\n' "$method" >&2; exit 2 ;;
    esac
done

# 数据和模型路径相对于仓库根目录解析，支持从任意目录启动。
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir"

for method in "${methods[@]}"; do
    case "$method" in
        rekv) args=("${rekv_args[@]}") ;;
        suprkv) args=("${suprkv_args[@]}") ;;
        streamkv) args=("${streamkv_args[@]}") ;;
        infinipot) args=("${infinipot_args[@]}") ;;
    esac
    command=(python -m video_qa.run_eval --method "$method" "${args[@]}" "$@")
    # 保留原 ReKV 脚本的 CUDA 设备排序设置，仅作用于该方法。
    if [[ "$method" == rekv ]]; then
        command=(env CUDA_DEVICE_ORDER=PCI_BUS_ID "${command[@]}")
    fi
    printf '\n>>> %s\n' "$method"
    printf ' %q' "${command[@]}"
    printf '\n'
    if [[ "$dry_run" == true ]]; then
        continue
    fi
    if "${command[@]}"; then
        printf '<<< %s 完成\n' "$method"
    else
        status=$?
        printf '方法 %s 失败（退出码 %s），停止后续执行。\n' "$method" "$status" >&2
        exit "$status"
    fi
done
