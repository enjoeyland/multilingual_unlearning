#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

world_size=2

python run.py \
    --model_name mt5-base \
    --model google/mt5-base \
    --method original \
    --cache_dir ../.cache \
    --task xnli \
    --max_length 512 \
    --forget_ratio 0.01 \
    --data_dir ../../research/multilingual-unlearning/data/ \
    --num_workers 4 \
    --do_train \
    --seed 42 \
    --dp_strategy fsdp \
    --bf16 \
    --optimizer adamw \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --epochs 3 \
    --world_size $world_size \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --logging_steps 200 \
    --eval_steps 500 \
    --max_tolerance 5 \
    --output_dir ".checkpoints/"