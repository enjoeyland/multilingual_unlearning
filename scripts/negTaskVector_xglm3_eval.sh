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
batch_size=16
dp_strategy="auto"


seed=42
learning_rate=("1e-5")
scaling_coef=("0.3 0" "0.2 0" "0.1 0" "0.08 0" "0.06 0" "0.04 0" "0.02 0" "1 0.7" "1 0.6" "1 0.5" "1 0.4")

for lr in "${learning_rate[@]}"; do
for sc in "${scaling_coef[@]}"; do
IFS=' ' read -r fsc rsc <<< "$sc"
echo "Learning Rate: $lr, Forget Scaling Coefficient: $fsc, Retain Scaling Coefficient: $rsc"
python run.py \
    --model_name xglm-2.9B \
    --model facebook/xglm-2.9B \
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
    --warmup_ratio 0 \
    --epochs 30 \
    --world_size $world_size \
    --per_device_batch_size $batch_size \
    --gradient_accumulation_steps $((16 / batch_size)) \
    --logging_steps 32 \
    --eval_steps 1 \
    --max_tolerance 5 \
    --output_dir ".checkpoints/" \
    --do_eval
done
done