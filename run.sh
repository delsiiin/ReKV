export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# The number of processes utilized for parallel evaluation.
# Normally, set it to the number of GPUs on your machine.
# Yet, llava_ov_72b needs 4x 80GB GPUs. So set num_chunks to num_gpus//4.
num_chunks=1

# Maximum video records across all workers; 0 evaluates the full dataset.
max_samples=5

# Supported model: llava_ov_0.5b llava_ov_7b llava_ov_72b video_llava_7b longva_7b
model=llava_ov_7b

# Supported dataset: qaego4d egoschema cgbench mlvu activitynet_qa rvs_ego rvs_movie
# MLVU has an extremely long video (~9hr). Remove it in the annotation file if your system doesn't have enough RAM.
dataset=rvs_ego

python -m video_qa.run_eval \
    --num_chunks $num_chunks \
    --max_samples $max_samples \
    --model ${model} \
    --dataset ${dataset} \
    --sample_fps 0.5 \
    --n_local 15000 \
    --retrieve_size 64