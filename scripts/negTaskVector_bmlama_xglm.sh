#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

method="negtaskvector"

task="bmlama53"
langs=("en" "fr" "es" "ar" "pt" "vi" "ca" "hi" "bn" "id")

world_size=1
batch_size=16

# learning_rate=("5e-4" "3e-4" "1e-4")
# scaling_coef=("0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")

# for sc in "${scaling_coef[@]}"; do
# for lr in "${learning_rate[@]}"; do
python run.py \
    --model_name xglm-564M \
    --model facebook/xglm-564M \
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
    --learning_rate 5e-4 \
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