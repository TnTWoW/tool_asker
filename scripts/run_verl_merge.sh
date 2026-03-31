#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: bash scripts/run_verl_merge.sh <actor_checkpoint_dir> <target_hf_dir>"
    exit 1
fi

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$1" \
  --target_dir "$2"
