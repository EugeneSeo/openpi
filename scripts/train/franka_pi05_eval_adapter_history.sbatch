#!/bin/bash
#SBATCH --job-name=pi05_franka_eval_hist
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_4090:1

set -euo pipefail

################################################################################
# DreamZero Eval Comparison: offline val evaluation for adapter_history
################################################################################
# - Evaluates the latest N compact adapter history checkpoints from a train-split
#   run on the validation split.
# - Default settings:
#     latest_n   = 10
#     val_batches = 20
#     batch_size = 16
# - Writes adapter_history_val_eval.json/.md into the checkpoint directory.
################################################################################

DREAMZERO_ROOT="${DREAMZERO_ROOT:-/cluster/project/cvg/students/eugseo/workspace/dreamzero}"
ROOT_DIR="${ROOT_DIR:-$DREAMZERO_ROOT/dreamdifferent/baseline/openpi}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$DREAMZERO_ROOT/.." && pwd)}"
DEFAULT_LOG_DIR="$WORKSPACE_ROOT/logs"
if [ -z "${LOG_DIR:-}" ] || [ "$LOG_DIR" = "/logs" ] || [ "$LOG_DIR" = "//logs" ]; then
  LOG_DIR="$DEFAULT_LOG_DIR"
fi

if [ ! -d "$ROOT_DIR" ]; then
  echo "ERROR: ROOT_DIR does not exist: $ROOT_DIR"
  echo "Set DREAMZERO_ROOT or ROOT_DIR explicitly before submitting."
  exit 1
fi

cd "$ROOT_DIR"

if [ -z "${CHECKPOINT_DIR:-}" ]; then
  echo "ERROR: CHECKPOINT_DIR must be set before submitting this sbatch script."
  echo "Example:"
  echo "  CHECKPOINT_DIR=/cluster/scratch/eugseo/openpi_checkpoints/pi05_franka_orca_bag_groceries/<exp_name> sbatch scripts/train/franka_pi05_eval_adapter_history.sbatch"
  exit 1
fi

CONFIG_NAME="${CONFIG_NAME:-pi05_franka_orca_bag_groceries}"
LATEST_N="${LATEST_N:-10}"
VAL_BATCHES="${VAL_BATCHES:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-1}"
RUN_NAME="${RUN_NAME:-eval_$(basename "$CHECKPOINT_DIR")_${SLURM_JOB_ID:-manual}}"
RUN_TMP_DIR="$ROOT_DIR/tmp/${RUN_NAME}"

mkdir -p "$LOG_DIR"
mkdir -p "$RUN_TMP_DIR/uv-cache" "$RUN_TMP_DIR/jax-cache"
mkdir -p /cluster/scratch/eugseo/hf_home /cluster/scratch/eugseo/hf_datasets_cache
mkdir -p /cluster/scratch/eugseo/lerobot_home/local
mkdir -p /cluster/scratch/eugseo/openpi_data

unset LEROBOT_HOME
export TMPDIR="$RUN_TMP_DIR"
export UV_CACHE_DIR="$RUN_TMP_DIR/uv-cache"
export JAX_COMPILATION_CACHE_DIR="$RUN_TMP_DIR/jax-cache"
export HF_LEROBOT_HOME=/cluster/scratch/eugseo/lerobot_home
export HF_HOME=/cluster/scratch/eugseo/hf_home
export HF_DATASETS_CACHE=/cluster/scratch/eugseo/hf_datasets_cache
export OPENPI_DATA_HOME=/cluster/scratch/eugseo/openpi_data

ln -sfn /cluster/scratch/eugseo/datasets/bag_groceries_communal \
  /cluster/scratch/eugseo/lerobot_home/local/bag_groceries_communal

uv run scripts/eval_adapter_history.py \
  --config-name "$CONFIG_NAME" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --latest-n "$LATEST_N" \
  --val-batches "$VAL_BATCHES" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS"

echo "Finished adapter history evaluation for: $CHECKPOINT_DIR"
echo "Results:"
echo "  $CHECKPOINT_DIR/adapter_history_val_eval.json"
echo "  $CHECKPOINT_DIR/adapter_history_val_eval.md"
