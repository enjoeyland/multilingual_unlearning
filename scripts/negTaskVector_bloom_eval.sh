#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

method="negtaskvector"

task="flores"
langs=("en" "fr" "es" "zh" "ar" "vi" "eu" "ur" "te" "sw")
max_length=256
# task="bmlama53"
# langs=("en" "fr" "es" "pt" "ar" "vi" "ca" "hi" "bn")
# max_length=32

world_size=1
batch_size=16

learning_rate=("3e-5" "5e-5" "1e-5")
warmup_ratio=("0")

scaling_coef=("1" "0.8" "0.6" "0.4")
# scaling_coef=("0.3" "0.2" "0.1" "0.08" "0.06" "0.04" "0.02")
# scaling_coef=("0.4" "0.5" "0.6" "0.7")

for wr in "${warmup_ratio[@]}"; do
for lr in "${learning_rate[@]}"; do
for sc in "${scaling_coef[@]}"; do
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
    --max_length $max_length \
    --num_workers 4 \
    --data_dir ../../research/multilingual-unlearning/data/ \
    --forget_scaling_coef $sc \
    --retain_scaling_coef 0 \
    --seed 42 \
    --wandb_mode disabled \
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
done