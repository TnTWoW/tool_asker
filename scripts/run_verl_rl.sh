#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: bash scripts/run_verl_rl.sh <n_gpus_per_node> <save_path> <actor_model_path>"
    exit 1
fi

N_GPUS="$1"
SAVE_PATH="$2"
MODEL_PATH="$3"
shift 3

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/verl/rl/train.parquet}"
VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/verl/rl/val.parquet}"
PROJECT_NAME="${PROJECT_NAME:-tool-asker-rl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-itge-ppo-qwen2.5-1.5b}"
ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-64}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-1024}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-256}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-16}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ACTOR_MICRO_BATCH_SIZE:-2}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.name="$ROLLOUT_NAME" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE:-1}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_BSZ:-4}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_BSZ:-4}" \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  critic.model.path="$MODEL_PATH" \
  critic.optim.lr="${CRITIC_LR:-1e-5}" \
  critic.ppo_micro_batch_size_per_gpu="${CRITIC_MICRO_BATCH_SIZE:-2}" \
  algorithm.kl_ctrl.kl_coef="${KL_COEF:-0.001}" \
  trainer.default_local_dir="$SAVE_PATH" \
  trainer.logger="${LOGGER:-console}" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.val_before_train=False \
  trainer.critic_warmup=0 \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ:-20}" \
  trainer.test_freq="${TEST_FREQ:-20}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-200}" \
  custom_reward_function.path="$PROJECT_ROOT/src/verl_reward.py" \
  custom_reward_function.name=compute_score \
  "$@"
