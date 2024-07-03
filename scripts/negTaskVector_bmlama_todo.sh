#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

method="negtaskvector"

task="bmlama53"
langs=("en" "fr" "es" "pt" "ar" "vi" "ca" "hi" "bn")
max_length=32


model_name="xglm-2.9B"
model="facebook/xglm-2.9B"
world_size=2
batch_size=8
warmup_ratio=0
dp_strategy="deepspeed_stage_2"

# seed=("0" "485")
seed=("42")
learning_rate=("3e-4" "5e-4" "1e-4")
fit_target=("forget" "retain")

todo=("xglm-2.9B facebook/xglm-2.9B 0 3e-5 forget" "xglm-2.9B facebook/xglm-2.9B 0 3e-5 retain" "xglm-2.9B facebook/xglm-2.9B 485 3e-5 forget" "xglm-2.9B facebook/xglm-2.9B 485 3e-5 retain")
# todo+=("bloom-3b bigscience/bloom-3b 0 1e-5 forget" "bloom-3b bigscience/bloom-3b 0 1e-5 retain" "bloom-3b bigscience/bloom-3b 485 1e-5 forget" "bloom-3b bigscience/bloom-3b 485 1e-5 retain")

for t in "${todo[@]}"; do
IFS=' ' read -r model_name model s lr ft <<< "$t"
echo "Running $method $task $s $lr $ft"
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
    --fit_target $ft \
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
    --gradient_accumulation_steps $((32 / batch_size / world_size)) \
    --logging_steps 32 \
    --eval_steps 1 \
    --max_tolerance 30 \
    --output_dir ".checkpoints/"
done
