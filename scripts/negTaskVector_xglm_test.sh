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
batch_size=32

seed="485"
lr="3e-4"
scaling_coef=("1 0.5" "0.2 0")

for sc in "${scaling_coef[@]}"; do
IFS=' ' read -r fsc rsc <<< "$sc"
echo "Forget Scaling Coefficient: $fsc, Retain Scaling Coefficient: $rsc"
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
    --max_length $max_length \
    --num_workers 4 \
    --data_dir ../../research/multilingual-unlearning/data/ \
    --forget_scaling_coef $fsc \
    --retain_scaling_coef $rsc \
    --seed $seed \
    --wandb_mode disabled \
    --dp_strategy auto \
    --bf16 \
    --optimizer adamw \
    --learning_rate $lr \
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
    --do_test
done