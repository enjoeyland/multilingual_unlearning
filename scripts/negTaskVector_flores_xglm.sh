#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

method="negtaskvector"

task="flores"
langs=("en" "fr" "es" "zh" "ar" "vi" "eu" "ur" "te" "sw")

model_name="xglm-564M"
world_size=1
batch_size=8
warmup_ratio=0.1
dp_strategy="auto"

# model_name="xglm-2.9B"
# world_size=2
# batch_size=1
# warmup_ratio=0
# dp_strategy="deepspeed_stage_3"

# scaling_coef=("0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")
# retain_multiplier=("4" "5")
seed=("42")

# for sc in "${scaling_coef[@]}"; do
# for rm in "${retain_multiplier[@]}"; do
for s in "${seed[@]}"; do
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
    --fit_target both \
    --forget_scaling_coef 1 \
    --retain_scaling_coef 0.5 \
    --do_train \
    --seed $s \
    --dp_strategy $dp_strategy \
    --bf16 \
    --optimizer adamw \
    --learning_rate 3e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio $warmup_ratio \
    --epochs 30 \
    --world_size $world_size \
    --per_device_batch_size $batch_size \
    --gradient_accumulation_steps $((32 / world_size / batch_size)) \
    --logging_steps 32 \
    --eval_steps 1 \
    --max_tolerance 10 \
    --output_dir ".checkpoints/"
done