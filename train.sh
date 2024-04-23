#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

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
    --train \
    --seed 42 \
    --bf16 \
    --optimizer adamw \
    --learning_rate 5e-5 \
    --epochs 5 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --logging_steps 500 \
    --eval_steps 500 \
    --output_dir "checkpoints/" \
    --load_best_model_at_end