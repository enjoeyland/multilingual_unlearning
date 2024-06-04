#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

method="negtaskvector"

task="flores"
langs=("en" "fr" "es" "zh" "ar" "vi" "eu" "ur" "te" "sw")

world_size=1
batch_size=8

# scaling_coef=("0.1" "0.05")

# for sc in "${scaling_coef[@]}"; do
python run.py \
    --model_name xglm-564M \
    --model facebook/xglm-564M \
    --method $method \
    --cache_dir ../.cache \
    --task ${task} \
    --forget_lang ${langs[@]} \
    --retain_lang ${langs[@]} \
    --forget_num 32 \
    --max_length 256 \
    --num_workers 4 \
    --data_dir ../../research/multilingual-unlearning/data/ \
    --scaling_coef 0.1 \
    --seed 42 \
    --dp_strategy auto \
    --bf16 \
    --optimizer adamw \
    --learning_rate 3e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --epochs 30 \
    --world_size $world_size \
    --per_device_batch_size $batch_size \
    --gradient_accumulation_steps $((32 / world_size / batch_size)) \
    --logging_steps 32 \
    --eval_steps 1 \
    --max_tolerance 5 \
    --output_dir ".checkpoints/" \
    --do_eval
# done