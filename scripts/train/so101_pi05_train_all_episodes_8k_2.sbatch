#!/bin/bash
#SBATCH --job-name=pi05_so101_all_8k
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=a100:1

set -euo pipefail

################################################################################
# SO101 Teleop Test Filtered: pi0.5-base LoRA training
################################################################################
# - Uses the local OpenPI-compatible converted LeRobot dataset:
#     $HF_LEROBOT_HOME/local/so101_teleop_test_filtered_openpi
# - Default run uses batch_size * num_train_steps = 16 * 500 = 8000,
#   matching the Franka/Grocery all-episodes 8k comparison budget.
# - Saves compact adapter checkpoints only by default to avoid multi-GB full
#   checkpoint churn on scratch/project storage.
# - Can resume an existing run by setting RESUME=1 and reusing EXP_NAME.
################################################################################

CONFIG_NAME="${CONFIG_NAME:-pi05_so101_teleop_test_filtered}"
DREAMZERO_ROOT="${DREAMZERO_ROOT:-/cluster/project/cvg/students/dohkim/dreamzero}"
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

EXP_NAME="${EXP_NAME:-so101_lora_all_episodes_bs16_8k_${SLURM_JOB_ID:-manual}}"
RUN_TMP_DIR="${RUN_TMP_DIR:-$ROOT_DIR/tmp/${EXP_NAME}}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
NUM_WORKERS="${NUM_WORKERS:-1}"
ADAPTER_HISTORY_TO_KEEP="${ADAPTER_HISTORY_TO_KEEP:-10}"
RESUME="${RESUME:-0}"
INIT_FROM_EXP_NAME="${INIT_FROM_EXP_NAME:-}"
INIT_FROM_CHECKPOINT_DIR="${INIT_FROM_CHECKPOINT_DIR:-}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"

DATASET_SRC="${DATASET_SRC:-/cluster/project/cvg/students/dohkim/datasets/local/so101_teleop_test_filtered_openpi}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/cluster/project/cvg/students/dohkim/datasets}"
HF_HOME="${HF_HOME:-$ROOT_DIR/tmp/hf-home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT_DIR/tmp/hf-datasets-cache}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$ROOT_DIR/tmp/openpi-data}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-$ROOT_DIR/assets}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/cluster/scratch/$USER/openpi_checkpoints}"

mkdir -p "$LOG_DIR"
mkdir -p "$RUN_TMP_DIR/uv-cache" "$RUN_TMP_DIR/jax-cache"
mkdir -p "$HF_LEROBOT_HOME/local" "$HF_HOME" "$HF_DATASETS_CACHE"
mkdir -p "$OPENPI_DATA_HOME" "$CHECKPOINT_BASE_DIR"

unset LEROBOT_HOME
export TMPDIR="$RUN_TMP_DIR"
export RUN_TMP_DIR
export UV_CACHE_DIR="$RUN_TMP_DIR/uv-cache"
export JAX_COMPILATION_CACHE_DIR="$RUN_TMP_DIR/jax-cache"
export HF_LEROBOT_HOME
export HF_HOME
export HF_DATASETS_CACHE
export OPENPI_DATA_HOME
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export PYTHONUNBUFFERED=1

if [ ! -d "$DATASET_SRC" ]; then
  echo "ERROR: converted dataset not found at $DATASET_SRC"
  echo "Run scripts/convert_so101_v3_to_openpi_lerobot.py first, or set DATASET_SRC."
  exit 1
fi

if [ ! -f "$ASSETS_BASE_DIR/$CONFIG_NAME/local/so101_teleop_test_filtered_openpi/norm_stats.json" ]; then
  echo "ERROR: missing norm stats:"
  echo "  $ASSETS_BASE_DIR/$CONFIG_NAME/local/so101_teleop_test_filtered_openpi/norm_stats.json"
  echo "Run scripts/compute_norm_stats.py --config-name $CONFIG_NAME before submitting."
  exit 1
fi

ln -sfn "$DATASET_SRC" "$HF_LEROBOT_HOME/local/so101_teleop_test_filtered_openpi"

if [ "$RESUME" = "1" ]; then
  RUN_MODE_FLAG="--resume"
else
  RUN_MODE_FLAG="--overwrite"
fi

if [ -n "$INIT_FROM_EXP_NAME" ] && [ -z "$INIT_FROM_CHECKPOINT_DIR" ]; then
  INIT_FROM_CHECKPOINT_DIR="$CHECKPOINT_BASE_DIR/$CONFIG_NAME/$INIT_FROM_EXP_NAME"
fi

EXTRA_ARGS=()
if [ -n "$INIT_FROM_CHECKPOINT_DIR" ]; then
  EXTRA_ARGS+=(--init-from-checkpoint-dir "$INIT_FROM_CHECKPOINT_DIR")
fi
if [ "$WANDB_ENABLED" = "1" ]; then
  EXTRA_ARGS+=(--wandb-enabled)
else
  EXTRA_ARGS+=(--no-wandb-enabled)
fi

echo "=== SO101 pi0.5 LoRA training ==="
echo "ROOT_DIR=$ROOT_DIR"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "EXP_NAME=$EXP_NAME"
echo "DATASET=$(readlink -f "$HF_LEROBOT_HOME/local/so101_teleop_test_filtered_openpi")"
echo "OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
echo "ASSETS_BASE_DIR=$ASSETS_BASE_DIR"
echo "CHECKPOINT_BASE_DIR=$CHECKPOINT_BASE_DIR"

uv run scripts/train.py "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --assets-base-dir "$ASSETS_BASE_DIR" \
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
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
echo "Checkpoint dir: $CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME"
