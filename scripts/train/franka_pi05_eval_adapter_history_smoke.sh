#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_NAME="${CONFIG_NAME:-pi05_franka_orca_bag_groceries}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/cluster/scratch/eugseo/openpi_checkpoints/pi05_franka_orca_bag_groceries/adapter_history_smoke_bs16}"
RUN_TMP_DIR="${RUN_TMP_DIR:-$ROOT_DIR/tmp}"

LATEST_N="${LATEST_N:-10}"
VAL_BATCHES="${VAL_BATCHES:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-1}"

mkdir -p "$RUN_TMP_DIR/uv-cache" "$RUN_TMP_DIR/jax-cache"

unset LEROBOT_HOME
export TMPDIR="$RUN_TMP_DIR"
export UV_CACHE_DIR="$RUN_TMP_DIR/uv-cache"
export JAX_COMPILATION_CACHE_DIR="$RUN_TMP_DIR/jax-cache"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/cluster/scratch/eugseo/lerobot_home}"
export HF_HOME="${HF_HOME:-/cluster/scratch/eugseo/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/cluster/scratch/eugseo/hf_datasets_cache}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/cluster/scratch/eugseo/openpi_data}"

uv run scripts/eval_adapter_history.py \
  --config-name "$CONFIG_NAME" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --latest-n "$LATEST_N" \
  --val-batches "$VAL_BATCHES" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS"
