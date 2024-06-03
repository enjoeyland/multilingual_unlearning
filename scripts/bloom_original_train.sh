#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

world_size=1
batch_size=64

python run.py \
    --model_name bloom-560M \
    --model bigscience/bloom-560M \
    --method original \
    --cache_dir ../.cache \
    --task xnli \
    --max_length 512 \
    --forget_ratio 0.01 \
    --data_dir ../../research/multilingual-unlearning/data/ \
    --num_workers 4 \
    --do_train \
    --use_lora \
    --load_in_4bit \
    --seed 42 \
    --dp_strategy auto \
    --bf16 \
    --optimizer adamw \
    --learning_rate 3e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --epochs 3 \
    --world_size $world_size \
    --per_device_batch_size $batch_size \
    --gradient_accumulation_steps $((128 / batch_size)) \
    --logging_steps $((1600 / batch_size)) \
    --eval_steps $((8000 / batch_size)) \
    --max_tolerance 15 \
    --output_dir ".checkpoints/"