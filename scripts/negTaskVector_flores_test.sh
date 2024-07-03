#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

method="negtaskvector"

task="flores"
langs=("en" "fr" "es" "zh" "ar" "vi" "eu" "ur" "te" "sw")
max_length=125
# task="bmlama53"
# langs=("en" "fr" "es" "pt" "ar" "vi" "ca" "hi" "bn")
# max_length=32

world_size=1
batch_size=32
dp_strategy="auto"

# todo=("xglm-2.9B facebook/xglm-2.9B 0 1e-4 1 0.6" "xglm-2.9B facebook/xglm-2.9B 485 1e-4 1 0.6")
todo=("bloom-3b bigscience/bloom-3b 0 1e-5 1 0.5" "bloom-3b bigscience/bloom-3b 485 1e-5 1 0.5")

for t in "${todo[@]}"; do
IFS=' ' read -r model_name model seed lr fsc rsc <<< "$t"
echo "Running $method $task $seed $lr"
echo "Forget Scaling Coefficient: $fsc, Retain Scaling Coefficient: $rsc"
python run.py \
    --model_name $model_name \
    --model $model \
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
    --dp_strategy $dp_strategy \
    --bf16 \
    --optimizer adamw \
    --learning_rate $lr \
    --lr_scheduler_type linear \
    --warmup_ratio 0 \
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