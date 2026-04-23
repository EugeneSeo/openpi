#!/usr/bin/env bash
set -euo pipefail

################################################################################
# DreamZero Eval Comparison: Franka/ORCA Bag-Groceries pi0.5-Base Training
################################################################################
# This script prepares the scratch-copied LeRobot v2 dataset for OpenPI and runs
# a pi0.5-base LoRA fine-tuning job with the DreamZero comparison settings:
#   - bimanual Franka + ORCA hand state/action, 48D
#   - 50Hz data, action_horizon=24
#   - relative action target: raw_action[t:t+24] - state[t]
#   - prompt from LeRobot task_index -> meta/tasks.jsonl
#
# The dataset itself should live on scratch:
#   /cluster/scratch/eugseo/datasets/bag_groceries_communal
################################################################################

CONFIG_NAME="${CONFIG_NAME:-pi05_franka_orca_bag_groceries}"
DATASET_SRC="${DATASET_SRC:-/cluster/scratch/eugseo/datasets/bag_groceries_communal}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/cluster/scratch/eugseo/lerobot_home}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/cluster/scratch/eugseo/openpi_data}"
HF_HOME="${HF_HOME:-/cluster/scratch/eugseo/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/cluster/scratch/eugseo/hf_datasets_cache}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-/cluster/scratch/eugseo/openpi_assets}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/cluster/scratch/eugseo/openpi_checkpoints}"
EXP_NAME="${EXP_NAME:-pi05_base_franka_orca_$(date +%Y%m%d_%H%M%S)}"
NORM_STATS_MAX_FRAMES="${NORM_STATS_MAX_FRAMES:-10000}"
SKIP_NORM_STATS="${SKIP_NORM_STATS:-0}"
RUN_TMP_DIR="${RUN_TMP_DIR:-$(pwd)/tmp}"

mkdir -p "$RUN_TMP_DIR/uv-cache" "$RUN_TMP_DIR/jax-cache" "$HF_HOME" "$HF_DATASETS_CACHE"

export TMPDIR="$RUN_TMP_DIR"
export UV_CACHE_DIR="$RUN_TMP_DIR/uv-cache"
export JAX_COMPILATION_CACHE_DIR="$RUN_TMP_DIR/jax-cache"
export HF_LEROBOT_HOME
export HF_HOME
export HF_DATASETS_CACHE
unset LEROBOT_HOME
export OPENPI_DATA_HOME
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

if [ ! -d "$DATASET_SRC" ]; then
  echo "ERROR: dataset not found at $DATASET_SRC"
  exit 1
fi

if [ ! -f "$DATASET_SRC/meta/tasks.jsonl" ]; then
  echo "ERROR: missing $DATASET_SRC/meta/tasks.jsonl"
  exit 1
fi

echo "=== Dataset task mapping ==="
sed -n '1,20p' "$DATASET_SRC/meta/tasks.jsonl"
if grep -q '"task"[[:space:]]*:[[:space:]]*"0"' "$DATASET_SRC/meta/tasks.jsonl"; then
  echo "ERROR: task text still appears to be \"0\". Fix meta/tasks.jsonl before training."
  exit 1
fi

mkdir -p "$HF_LEROBOT_HOME/local"
ln -sfn "$DATASET_SRC" "$HF_LEROBOT_HOME/local/bag_groceries_communal"

echo "=== OpenPI LeRobot path ==="
readlink -f "$HF_LEROBOT_HOME/local/bag_groceries_communal"

if [ "$SKIP_NORM_STATS" != "1" ]; then
  echo "=== Computing OpenPI norm stats after Franka/ORCA relative-action transforms ==="
  uv run scripts/compute_norm_stats.py \
    --config-name "$CONFIG_NAME" \
    --max-frames "$NORM_STATS_MAX_FRAMES"

  mkdir -p "$ASSETS_BASE_DIR/$CONFIG_NAME/local"
  rsync -a "assets/$CONFIG_NAME/local/bag_groceries_communal/" \
    "$ASSETS_BASE_DIR/$CONFIG_NAME/local/bag_groceries_communal/"
else
  echo "=== Skipping norm stats computation because SKIP_NORM_STATS=1 ==="
fi

echo "=== Starting pi0.5-base Franka/ORCA LoRA training ==="
uv run scripts/train.py "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --assets-base-dir "$ASSETS_BASE_DIR" \
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
  --overwrite
