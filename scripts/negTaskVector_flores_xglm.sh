#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

method="negtaskvector"

task="flores"
langs=("en" "fr" "es" "zh" "ar" "vi" "eu" "ur" "te" "sw")

# model_name="xglm-564M"
# world_size=1
# batch_size=8
# warmup_ratio=0.1
# dp_strategy="auto"
# max_length=256

model_name="xglm-2.9B"
world_size=2
batch_size=4
warmup_ratio=0
dp_strategy="deepspeed_stage_2"
max_length=125

seed=("42")
learning_rate=("3e-5")
# learning_rate=("3e-5" "5e-5" "1e-4")
fit_target=("forget" "retain")
# fit_target=("forget")

for s in "${seed[@]}"; do
for lr in "${learning_rate[@]}"; do
for ft in "${fit_target[@]}"; do
echo "Running $method $task $s $lr $ft"
python run.py \
    --model_name $model_name \
    --model "facebook/$model_name" \
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
    --fit_target $ft \
    --forget_scaling_coef 1 \
    --retain_scaling_coef 0.5 \
    --do_train \
    --seed $s \
    --dp_strategy $dp_strategy \
    --bf16 \
    --optimizer adamw \
    --learning_rate $lr \
    --lr_scheduler_type linear \
    --warmup_ratio $warmup_ratio \
    --epochs 30 \
    --world_size $world_size \
    --per_device_batch_size $batch_size \
    --gradient_accumulation_steps $((32 / batch_size)) \
    --logging_steps 32 \
    --eval_steps 1 \
    --max_tolerance 10 \
    --output_dir ".checkpoints/"
done
done
done