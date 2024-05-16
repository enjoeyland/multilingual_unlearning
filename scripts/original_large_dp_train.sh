#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

world_size=2

python run.py \
    --model_name bloomz-560m \
    --model bigscience/bloomz-560m \
    --method original \
    --cache_dir ../.cache \
    --task xnli \
    --max_length 512 \
    --forget_ratio 0.01 \
    --data_dir ../../research/multilingual-unlearning/data/ \
    --num_workers 4 \
    --do_train \
    --seed 42 \
    --dp_strategy deepspeed_stage_3_offload \
    --bf16 \
    --optimizer adamw \
    --learning_rate 3e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --epochs 3 \
    --world_size $world_size \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --logging_steps 200 \
    --eval_steps 1000 \
    --max_tolerance 15 \
    --output_dir ".checkpoints/"
