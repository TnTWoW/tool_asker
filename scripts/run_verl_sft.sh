#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: bash scripts/run_verl_sft.sh <nproc_per_node> <save_path> [model_path]"
    exit 1
fi

NPROC_PER_NODE="$1"
SAVE_PATH="$2"
MODEL_PATH="${3:-Qwen/Qwen2.5-1.5B-Instruct}"
if [ "$#" -ge 3 ]; then
  shift 3
else
  shift 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/verl/sft/train.parquet}"
VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/verl/sft/val.parquet}"
PROJECT_NAME="${PROJECT_NAME:-tool-asker-sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-itge-sft-qwen2.5-1.5b}"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.prompt_key=prompt \
  data.response_key=response \
  data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE:-4}" \
  data.max_length="${MAX_LENGTH:-2048}" \
  optim.lr="${LR:-1e-4}" \
  model.partial_pretrain="$MODEL_PATH" \
  model.enable_gradient_checkpointing=True \
  model.lora_rank="${LORA_RANK:-32}" \
  model.lora_alpha="${LORA_ALPHA:-16}" \
  model.target_modules=all-linear \
  trainer.default_local_dir="$SAVE_PATH" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger="${LOGGER:-console}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-2}" \
  trainer.test_freq="${TEST_FREQ:-1}" \
  trainer.n_gpus_per_node="$NPROC_PER_NODE" \
  trainer.nnodes=1 \
  "$@"
