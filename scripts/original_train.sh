#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

world_size=1

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
    --bf16 \
    --optimizer adamw \
    --learning_rate 5e-5 \
    --epochs 3 \
    --world_size $world_size \
    --batch_size 8 \
    --gradient_accumulation_steps $((4 / world_size)) \
    --logging_steps $((200 / world_size)) \
    --eval_steps $((500 / world_size)) \
    --max_tolerance 5 \
    --output_dir ".checkpoints/"