#!/usr/bin/env bash
set -euo pipefail

# Required environment variables:
#   PRETRAINED_MODEL  Hugging Face model name or local directory
#   TRAIN_DATA        Prepared Arrow dataset directory
# Optional:
#   RUN_NAME, OUTPUT_ROOT, NPROC, SEED, MASTER_PORT

: "${PRETRAINED_MODEL:?Set PRETRAINED_MODEL to a model name or directory}"
: "${TRAIN_DATA:?Set TRAIN_DATA to the prepared Arrow dataset directory}"

RUN_NAME="${RUN_NAME:-decir-safe-seed42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./runs}"
NPROC="${NPROC:-2}"
SEED="${SEED:-42}"
MASTER_PORT="${MASTER_PORT:-30013}"

mkdir -p \
    "${OUTPUT_ROOT}/models/${RUN_NAME}" \
    "${OUTPUT_ROOT}/tensorboard/${RUN_NAME}" \
    "${OUTPUT_ROOT}/configs"

colossalai run --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT}" train.py \
    --pretrained "${PRETRAINED_MODEL}" \
    --dataset "${TRAIN_DATA}" \
    --plugin zero2 \
    --save_interval 400 \
    --save_dir "${OUTPUT_ROOT}/models/${RUN_NAME}" \
    --tensorboard_dir "${OUTPUT_ROOT}/tensorboard/${RUN_NAME}" \
    --config_file "${OUTPUT_ROOT}/configs/${RUN_NAME}.json" \
    --num_epochs 2 \
    --micro_batch_size 8 \
    --accumulation_steps 8 \
    --lr 2e-5 \
    --mixed_precision bf16 \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --use_grad_checkpoint \
    --padding_mode longest \
    --max_length 4096 \
    --use_flash_attn \
    --pad_token eos \
    --seed "${SEED}"
