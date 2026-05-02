# Franka/ORCA Training Scripts

These scripts were written for the ETH Euler cluster environment.

In particular, they assume:

- Slurm jobs on Euler.
- Scratch paths under `/cluster/scratch/eugseo/`.
- An OpenPI baseline checkout at `dreamdifferent/baseline/openpi/` inside a
  larger project workspace.
- OpenPI temporary/cache directories under `baseline/openpi/tmp/`.
- A local LeRobot dataset symlink at:
  `/cluster/scratch/eugseo/lerobot_home/local/bag_groceries_communal`.
- Saved split metadata and dataset analysis under
  `dreamdifferent/datasets/bag_groceries_communal/`.

If you run them on another machine or under another account, update the paths
and resource requests accordingly.

The sbatch files intentionally do not hardcode a Slurm output/error path. A
common pattern is to submit them with an explicit DreamZero root and log
directory, for example:

```bash
DREAMZERO_ROOT=/cluster/project/cvg/students/eugseo/workspace/dreamzero
OPENPI_ROOT="$DREAMZERO_ROOT/dreamdifferent/baseline/openpi"
WORKSPACE_ROOT="$(cd "$DREAMZERO_ROOT/.." && pwd)"
LOG_DIR="$WORKSPACE_ROOT/logs"
mkdir -p "$LOG_DIR"
```

## First-Time Submodule Setup

If this OpenPI checkout was pulled in as a fresh submodule, prepare its local
environment once on a login node before launching `sbatch` jobs:

```bash
DREAMZERO_ROOT=/cluster/project/cvg/students/eugseo/workspace/dreamzero
OPENPI_ROOT="$DREAMZERO_ROOT/dreamdifferent/baseline/openpi"

cd "$OPENPI_ROOT"
mkdir -p "$OPENPI_ROOT/tmp/uv-cache"
UV_CACHE_DIR="$OPENPI_ROOT/tmp/uv-cache" uv sync
```

This matters because the Euler compute nodes may not have outbound GitHub
access. If the submodule does not already have a prepared `.venv`, `uv run`
inside the job can fail while trying to fetch the `lerobot` dependency.

You can sanity-check the environment with:

```bash
cd "$OPENPI_ROOT"
UV_CACHE_DIR="$OPENPI_ROOT/tmp/uv-cache" uv pip show lerobot
```

If the OpenPI submodule dependencies change later, rerun `uv sync` from the
same directory.

If the bag-groceries dataset is not already present on your Euler scratch, copy
it first from the ETH student cluster:

```bash
mkdir -p /cluster/scratch/$USER/datasets/bag_groceries_communal

rsync -aH --partial --info=progress2 \
  <eth-username>@student-cluster1.inf.ethz.ch:/work/courses/3dv/team21/datasets/bag_groceries_communal/ \
  /cluster/scratch/$USER/datasets/bag_groceries_communal/

mkdir -p /cluster/scratch/$USER/lerobot_home/local

ln -sfn /cluster/scratch/$USER/datasets/bag_groceries_communal \
  /cluster/scratch/$USER/lerobot_home/local/bag_groceries_communal
```

## Script Overview

### `franka_pi05_base_training.sh`

General helper for preparing the Franka/ORCA bag-groceries dataset and running
the `pi05_franka_orca_bag_groceries` config.

Use this when you want a simple shell entrypoint and may want to override:

- dataset source
- norm-stat computation
- asset/checkpoint directories
- experiment name

### `franka_pi05_train_split_only_smoke.sh`

Small smoke test on the saved train split only.

Behavior:

- uses the saved 90/10 episode split
- disables validation during training
- saves compact adapter checkpoints only
- keeps the latest 10 adapter-history checkpoints

### `franka_pi05_train_all_episodes_smoke.sh`

Small smoke test on all 300 episodes.

Behavior:

- disables the saved split with `--data.split-path None`
- disables validation during training
- saves compact adapter checkpoints only
- keeps the latest 10 adapter-history checkpoints

### `franka_pi05_eval_adapter_history_smoke.sh`

Offline validation helper for a completed train-split run.

It evaluates the latest `N` checkpoints from `adapter_history/` on the
validation split and writes:

- `adapter_history_val_eval.json`
- `adapter_history_val_eval.md`

Use this after a train-split run when you want to pick the best checkpoint from
the latest window.

### `franka_pi05_train_split_only_8k.sbatch`

Main Euler `sbatch` job for train-split-only training.

Default setup:

- `batch_size=16`
- `num_train_steps=500`
- `batch_size * num_train_steps = 8000`
- `action_horizon=24` from `pi05_franka_orca_bag_groceries`
- compact adapter checkpoints only
- rolling `adapter_history` with the latest 10 checkpoints

This script now supports three modes:

1. Fresh run:
   - default behavior
   - creates a new experiment directory

2. Resume in the same experiment directory:
   - set `RESUME=1`
   - reuse the same `EXP_NAME`
   - increase `NUM_TRAIN_STEPS` to the new total target

3. Branch from an existing checkpoint into a new experiment directory:
   - keep `RESUME=0`
   - choose a new `EXP_NAME`
   - set `INIT_FROM_EXP_NAME=<source_exp_name>`

### `franka_pi05_train_all_episodes_8k.sbatch`

Main Euler `sbatch` job for all-episodes training.

Same logic as the train-split script, except it disables the saved split and
trains on all 300 episodes.

This is closer to a latest-checkpoint robotics evaluation setup.

### `franka_pi05_eval_adapter_history.sbatch`

Euler `sbatch` job for offline validation over `adapter_history/`.

Set:

- `CHECKPOINT_DIR=<full_checkpoint_dir>`

before submission. By default it evaluates the latest 10 checkpoints with
`VAL_BATCHES=20`.

## Typical Usage

### Train split only, fresh run

```bash
DREAMZERO_ROOT=/cluster/project/cvg/students/eugseo/workspace/dreamzero
OPENPI_ROOT="$DREAMZERO_ROOT/dreamdifferent/baseline/openpi"
WORKSPACE_ROOT="$(cd "$DREAMZERO_ROOT/.." && pwd)"
LOG_DIR="$WORKSPACE_ROOT/logs"
mkdir -p "$LOG_DIR"

cd "$OPENPI_ROOT"
DREAMZERO_ROOT="$DREAMZERO_ROOT" \
sbatch \
  --output "$LOG_DIR/%x-%j.out" \
  --error "$LOG_DIR/%x-%j.err" \
  scripts/train/franka_pi05_train_split_only_8k.sbatch
```

### All episodes, fresh run

```bash
DREAMZERO_ROOT=/cluster/project/cvg/students/eugseo/workspace/dreamzero
OPENPI_ROOT="$DREAMZERO_ROOT/dreamdifferent/baseline/openpi"
WORKSPACE_ROOT="$(cd "$DREAMZERO_ROOT/.." && pwd)"
LOG_DIR="$WORKSPACE_ROOT/logs"
mkdir -p "$LOG_DIR"

cd "$OPENPI_ROOT"
DREAMZERO_ROOT="$DREAMZERO_ROOT" \
sbatch \
  --output "$LOG_DIR/%x-%j.out" \
  --error "$LOG_DIR/%x-%j.err" \
  scripts/train/franka_pi05_train_all_episodes_8k.sbatch
```

### Resume an existing run

```bash
DREAMZERO_ROOT=/cluster/project/cvg/students/eugseo/workspace/dreamzero
OPENPI_ROOT="$DREAMZERO_ROOT/dreamdifferent/baseline/openpi"
WORKSPACE_ROOT="$(cd "$DREAMZERO_ROOT/.." && pwd)"
LOG_DIR="$WORKSPACE_ROOT/logs"
mkdir -p "$LOG_DIR"

cd "$OPENPI_ROOT"
DREAMZERO_ROOT="$DREAMZERO_ROOT" \
EXP_NAME=<existing_exp_name> \
RESUME=1 \
NUM_TRAIN_STEPS=1000 \
sbatch \
  --output "$LOG_DIR/%x-%j.out" \
  --error "$LOG_DIR/%x-%j.err" \
  scripts/train/franka_pi05_train_split_only_8k.sbatch
```

`NUM_TRAIN_STEPS` is the final total step target, not the increment.

### Branch from an existing run

```bash
DREAMZERO_ROOT=/cluster/project/cvg/students/eugseo/workspace/dreamzero
OPENPI_ROOT="$DREAMZERO_ROOT/dreamdifferent/baseline/openpi"
WORKSPACE_ROOT="$(cd "$DREAMZERO_ROOT/.." && pwd)"
LOG_DIR="$WORKSPACE_ROOT/logs"
mkdir -p "$LOG_DIR"

cd "$OPENPI_ROOT"
DREAMZERO_ROOT="$DREAMZERO_ROOT" \
EXP_NAME=<new_exp_name> \
INIT_FROM_EXP_NAME=<source_exp_name> \
NUM_TRAIN_STEPS=1000 \
sbatch \
  --output "$LOG_DIR/%x-%j.out" \
  --error "$LOG_DIR/%x-%j.err" \
  scripts/train/franka_pi05_train_split_only_8k.sbatch
```

This preserves the source run and starts a new run from its `adapter_latest/`
checkpoint.

### Evaluate the latest checkpoint window on the validation split

```bash
DREAMZERO_ROOT=/cluster/project/cvg/students/eugseo/workspace/dreamzero
OPENPI_ROOT="$DREAMZERO_ROOT/dreamdifferent/baseline/openpi"
WORKSPACE_ROOT="$(cd "$DREAMZERO_ROOT/.." && pwd)"
LOG_DIR="$WORKSPACE_ROOT/logs"
mkdir -p "$LOG_DIR"

cd "$OPENPI_ROOT"
DREAMZERO_ROOT="$DREAMZERO_ROOT" \
CHECKPOINT_DIR=/cluster/scratch/eugseo/openpi_checkpoints/pi05_franka_orca_bag_groceries/<exp_name> \
sbatch \
  --output "$LOG_DIR/%x-%j.out" \
  --error "$LOG_DIR/%x-%j.err" \
  scripts/train/franka_pi05_eval_adapter_history.sbatch
```

## Notes

- These scripts are Euler-oriented convenience wrappers, not general OpenPI
  upstream scripts.
- They assume the custom config `pi05_franka_orca_bag_groceries` exists in this
  checkout.
- The train loss logged by OpenPI here is the pi0.5 flow-matching loss on
  normalized relative-action chunks, not direct raw-action MSE.

## SO101 Absolute vs Delta Action Note

The current SO101 config, `pi05_so101_teleop_test_filtered`, trains on the
dataset action values as-is. In other words, the model is trained to predict the
same 6D action convention stored in the converted SO101 LeRobot dataset, and
the policy server returns a `(24, 6)` chunk of absolute SO101 actions after
normalization is undone.

This is different from the Franka/ORCA comparison config, which explicitly
converts actions to relative deltas during training and converts them back to
absolute actions during inference with `DeltaActions` / `AbsoluteActions`.

If a future SO101 experiment should train the model to predict deltas, update
`LeRobotSO101DataConfig.create()` in `src/openpi/training/config.py` so the
SO101 data transform stack mirrors the Franka pattern:

```python
relative_action_mask = (True,) * so101_policy.SO101_ACTION_DIM
data_transforms = _transforms.Group(
    inputs=[so101_policy.SO101Inputs(model_type=model_config.model_type)],
    outputs=[so101_policy.SO101Outputs()],
).push(
    inputs=[_transforms.DeltaActions(relative_action_mask)],
    outputs=[_transforms.AbsoluteActions(relative_action_mask)],
)
```

With this setup, training targets become `action - current_state`, so the model
internally learns delta actions. Because `AbsoluteActions` is present in the
output transform stack, the standard policy server will still return absolute
actions to the robot client (`predicted_delta + current_state`).

If the robot client should receive raw deltas instead, train with
`DeltaActions` but do not include `AbsoluteActions` in the inference output
transform stack. This should be done as a separate SO101 delta config or serving
path, rather than reinterpreting an absolute-action checkpoint as delta. Existing
SO101 absolute-action checkpoints should not be treated as delta policies; they
must be retrained with the delta transform for that behavior.
