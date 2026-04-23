#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_NAME="${CONFIG_NAME:-pi05_franka_orca_bag_groceries}"
EXP_NAME="${EXP_NAME:-adapter_history_all_episodes_smoke_bs16}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/cluster/scratch/eugseo/openpi_checkpoints}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-$ROOT_DIR/assets}"
RUN_TMP_DIR="${RUN_TMP_DIR:-$ROOT_DIR/tmp}"

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
ADAPTER_HISTORY_TO_KEEP="${ADAPTER_HISTORY_TO_KEEP:-10}"

mkdir -p "$RUN_TMP_DIR/uv-cache" "$RUN_TMP_DIR/jax-cache"

unset LEROBOT_HOME
export TMPDIR="$RUN_TMP_DIR"
export UV_CACHE_DIR="$RUN_TMP_DIR/uv-cache"
export JAX_COMPILATION_CACHE_DIR="$RUN_TMP_DIR/jax-cache"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/cluster/scratch/eugseo/lerobot_home}"
export HF_HOME="${HF_HOME:-/cluster/scratch/eugseo/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/cluster/scratch/eugseo/hf_datasets_cache}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/cluster/scratch/eugseo/openpi_data}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

uv run scripts/train.py "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --assets-base-dir "$ASSETS_BASE_DIR" \
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
  --data.split-path None \
  --num-train-steps "$NUM_TRAIN_STEPS" \
  --log-interval "$LOG_INTERVAL" \
  --save-interval "$SAVE_INTERVAL" \
  --val-interval None \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --no-save-full-checkpoints \
  --no-save-full-best-val-checkpoint \
  --no-log-wandb-images \
  --adapter-history-to-keep "$ADAPTER_HISTORY_TO_KEEP" \
  --overwrite
