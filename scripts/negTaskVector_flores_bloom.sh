#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

method="negtaskvector"

task="flores"
langs=("en" "fr" "es" "zh" "ar" "vi" "eu" "ur" "te" "sw")

world_size=1
batch_size=8

learning_rate=("5e-5" "3e-5" "1e-5")
warmup_ratio=("0" "0.1")
# scaling_coef=("0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")
# retain_multiplier=("4" "5")

# for sc in "${scaling_coef[@]}"; do
# for rm in "${retain_multiplier[@]}"; do
for wr in "${warmup_ratio[@]}"; do
for lr in "${learning_rate[@]}"; do
python run.py \
    --model_name bloom-560m \
    --model bigscience/bloom-560m \
    --method $method \
    --cache_dir ../.cache \
    --task $task \
    --forget_lang ${langs[@]} \
    --retain_lang ${langs[@]} \
    --forget_num 32 \
    --retain_multiplier 1 \
    --max_length 256 \
    --num_workers 4 \
    --data_dir ../../research/multilingual-unlearning/data/ \
    --negtv_fit both \
    --forget_scaling_coef 0.08 \
    --retain_scaling_coef 0 \
    --do_train \
    --seed 42 \
    --dp_strategy auto \
    --bf16 \
    --optimizer adamw \
    --learning_rate $lr \
    --lr_scheduler_type linear \
    --warmup_ratio $wr \
    --epochs 30 \
    --world_size $world_size \
    --per_device_batch_size $batch_size \
    --gradient_accumulation_steps $((32 / world_size / batch_size)) \
    --logging_steps 32 \
    --eval_steps 1 \
    --max_tolerance 5 \
    --output_dir ".checkpoints/" \
    --do_eval
done
done