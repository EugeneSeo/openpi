#!/bin/bash
#SBATCH --job-name=pi05_franka_split_8k
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_4090:1

set -euo pipefail

################################################################################
# DreamZero Eval Comparison: pi0.5-base training on train split only
################################################################################
# - Uses the 270-episode train split from bag_groceries_seed42_90_10.json.
# - Keeps DreamZero-matched action_horizon=24 via the config.
# - Default run uses batch_size * num_train_steps = 16 * 500 = 8000.
# - Can also resume an existing run by setting RESUME=1 and reusing EXP_NAME.
# - Writes only compact adapter checkpoints and keeps the latest 10 history
#   checkpoints for offline validation-based selection after training.
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT_DIR/../../../.." && pwd)"
LOG_DIR="${LOG_DIR:-$WORKSPACE_ROOT/logs}"
cd "$ROOT_DIR"

EXP_NAME="${EXP_NAME:-pi05_train_split_bs16_8k_${SLURM_JOB_ID:-manual}}"
RUN_TMP_DIR="$ROOT_DIR/tmp/${EXP_NAME}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
ADAPTER_HISTORY_TO_KEEP="${ADAPTER_HISTORY_TO_KEEP:-10}"
RESUME="${RESUME:-0}"
INIT_FROM_EXP_NAME="${INIT_FROM_EXP_NAME:-}"
INIT_FROM_CHECKPOINT_DIR="${INIT_FROM_CHECKPOINT_DIR:-}"

mkdir -p "$LOG_DIR"
mkdir -p "$RUN_TMP_DIR/uv-cache" "$RUN_TMP_DIR/jax-cache"
mkdir -p /cluster/scratch/eugseo/hf_home /cluster/scratch/eugseo/hf_datasets_cache
mkdir -p /cluster/scratch/eugseo/lerobot_home/local
mkdir -p /cluster/scratch/eugseo/openpi_data

unset LEROBOT_HOME
export TMPDIR="$RUN_TMP_DIR"
export RUN_TMP_DIR
export UV_CACHE_DIR="$RUN_TMP_DIR/uv-cache"
export JAX_COMPILATION_CACHE_DIR="$RUN_TMP_DIR/jax-cache"
export HF_LEROBOT_HOME=/cluster/scratch/eugseo/lerobot_home
export HF_HOME=/cluster/scratch/eugseo/hf_home
export HF_DATASETS_CACHE=/cluster/scratch/eugseo/hf_datasets_cache
export OPENPI_DATA_HOME=/cluster/scratch/eugseo/openpi_data
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

if [ "$RESUME" = "1" ]; then
  RUN_MODE_FLAG="--resume"
else
  RUN_MODE_FLAG="--overwrite"
fi

if [ -n "$INIT_FROM_EXP_NAME" ] && [ -z "$INIT_FROM_CHECKPOINT_DIR" ]; then
  INIT_FROM_CHECKPOINT_DIR="/cluster/scratch/eugseo/openpi_checkpoints/pi05_franka_orca_bag_groceries/$INIT_FROM_EXP_NAME"
fi

EXTRA_ARGS=()
if [ -n "$INIT_FROM_CHECKPOINT_DIR" ]; then
  EXTRA_ARGS+=(--init-from-checkpoint-dir "$INIT_FROM_CHECKPOINT_DIR")
fi

ln -sfn /cluster/scratch/eugseo/datasets/bag_groceries_communal \
  /cluster/scratch/eugseo/lerobot_home/local/bag_groceries_communal

uv run scripts/train.py pi05_franka_orca_bag_groceries \
  --exp-name "$EXP_NAME" \
  --assets-base-dir "$ROOT_DIR/assets" \
  --checkpoint-base-dir /cluster/scratch/eugseo/openpi_checkpoints \
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
  "${EXTRA_ARGS[@]}" \
  "$RUN_MODE_FLAG"

echo "Finished run: $EXP_NAME"
echo "Checkpoint dir: /cluster/scratch/eugseo/openpi_checkpoints/pi05_franka_orca_bag_groceries/$EXP_NAME"
