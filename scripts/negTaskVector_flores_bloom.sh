#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

method="negtaskvector"

task="flores"
langs=("en" "fr" "es" "zh" "ar" "vi" "eu" "ur" "te" "sw")

world_size=1
batch_size=8

seed=("42")
learning_rate=("5e-5")
warmup_ratio=("0")

for s in "${seed[@]}"; do
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
    --fit_target both \
    --do_train \
    --seed $s \
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
    --output_dir ".checkpoints/"
done
done
done