#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

world_size=1


python run.py \
    --model_name mt5-base \
    --model google/mt5-base \
    --method sisa \
    --cache_dir ../.cache \
    --task xnli \
    --max_length 512 \
    --forget_ratio 0.01 \
    --data_dir ../../research/multilingual-unlearning/data/ \
    --num_workers 4 \
    --shards 5 \
    --slices 9 \
    --seed 42 \
    --dp_strategy auto \
    --bf16 \
    --optimizer adamw \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --epochs 5 \
    --world_size $world_size \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps $((4 / world_size)) \
    --logging_steps $((200 / world_size)) \
    --eval_steps $((500 / world_size)) \
    --max_tolerance 3 \
    --output_dir ".checkpoints/" \
    --do_eval