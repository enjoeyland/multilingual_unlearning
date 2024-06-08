#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

method="negtaskvector"

task="flores"
langs=("en" "fr" "es" "zh" "ar" "vi" "eu" "ur" "te" "sw")

world_size=1
batch_size=32

retain_multiplier=("1")
scaling_coef=("0.3" "0.2" "0.1" "0.08" "0.06" "0.04" "0.02")

for rm in "${retain_multiplier[@]}"; do
for sc in "${scaling_coef[@]}"; do
echo "Retain Multiplier: $rm, Scaling Coefficient: $sc"
python run.py \
    --model_name xglm-564M \
    --model facebook/xglm-564M \
    --method $method \
    --cache_dir ../.cache \
    --task $task \
    --forget_lang ${langs[@]} \
    --retain_lang ${langs[@]} \
    --forget_num 32 \
    --retain_multiplier $rm \
    --max_length 256 \
    --num_workers 4 \
    --data_dir ../../research/multilingual-unlearning/data/ \
    --forget_scaling_coef $sc \
    --retain_scaling_coef 0 \
    --seed 42 \
    --wandb_mode disabled \
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
done
done