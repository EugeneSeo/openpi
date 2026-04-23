from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import json
import logging
import shutil
from typing import Protocol

from etils import epath
import flax.nnx as nnx
import flax.traverse_util
import jax
import numpy as np
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

import openpi.models.model as _model
from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def save_best_val_state(
    directory: epath.Path | str,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    val_loss: float,
) -> None:
    """Synchronously overwrite a standalone best-validation checkpoint directory."""
    directory = epath.Path(directory).resolve()
    tmp_directory = directory.parent / f"{directory.name}.tmp"

    if tmp_directory.exists():
        tmp_directory.rmtree()
    tmp_directory.mkdir(parents=True, exist_ok=True)

    manager = ocp.CheckpointManager(
        tmp_directory,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(max_to_keep=1, create=False),
    )
    save_state(manager, state, data_loader, step)
    manager.wait_until_finished()
    saved_steps = tuple(manager.all_steps())
    manager.close()

    if not saved_steps:
        raise RuntimeError(f"Best-val checkpoint save did not create a step under {tmp_directory}")
    saved_step = saved_steps[-1]
    saved_step_dir = tmp_directory / str(saved_step)
    if not saved_step_dir.exists():
        step_dirs = [p for p in tmp_directory.iterdir() if p.is_dir()]
        if len(step_dirs) != 1:
            raise RuntimeError(f"Could not identify saved best-val step directory under {tmp_directory}")
        saved_step_dir = step_dirs[0]
    (saved_step_dir / "metadata.json").write_text(json.dumps({"step": step, "val_loss": val_loss}, indent=2) + "\n")

    if directory.exists():
        directory.rmtree()
    shutil.move(str(saved_step_dir), str(directory))
    tmp_directory.rmtree()


def save_adapter_state(
    directory: epath.Path | str,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    adapter_filter: nnx.filterlib.Filter,
    *,
    include_train_state: bool = False,
    metadata: dict[str, object] | None = None,
) -> None:
    ################################################################################
    # DreamZero Eval Comparison: compact adapter checkpoint
    ################################################################################
    # This writes only the trainable adapter subset plus assets. Optionally, it
    # also writes step + optimizer state without frozen base params, making
    # adapter_latest a minimal resumable checkpoint.
    ################################################################################

    def save_assets(directory: epath.Path):
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    directory = epath.Path(directory).resolve()
    tmp_directory = directory.parent / f"{directory.name}.tmp"

    if tmp_directory.exists():
        tmp_directory.rmtree()
    tmp_directory.mkdir(parents=True, exist_ok=True)

    with at.disable_typechecking():
        _, params = _split_params(state)
        adapter_params = params.filter(adapter_filter)
    adapter_leaf_count = len(flax.traverse_util.flatten_dict(adapter_params.to_pure_dict(), sep="/"))
    if adapter_leaf_count == 0:
        raise RuntimeError("Adapter checkpoint filter matched no parameter leaves.")

    manager = ocp.CheckpointManager(
        tmp_directory,
        item_handlers={
            "assets": CallbackHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(max_to_keep=1, create=False),
    )
    items = {
        "assets": save_assets,
        "params": {"params": adapter_params},
    }
    if include_train_state:
        if state.ema_params is not None:
            raise NotImplementedError("Minimal adapter resume checkpoints do not support EMA params yet.")
        with at.disable_typechecking():
            items["train_state"] = dataclasses.replace(state, params={}, ema_params=None)

    manager.save(step, items)
    manager.wait_until_finished()
    saved_steps = tuple(manager.all_steps())
    manager.close()

    if not saved_steps:
        raise RuntimeError(f"Adapter checkpoint save did not create a step under {tmp_directory}")
    saved_step = saved_steps[-1]
    saved_step_dir = tmp_directory / str(saved_step)
    if not saved_step_dir.exists():
        step_dirs = [p for p in tmp_directory.iterdir() if p.is_dir()]
        if len(step_dirs) != 1:
            raise RuntimeError(f"Could not identify saved adapter step directory under {tmp_directory}")
        saved_step_dir = step_dirs[0]

    metadata = {
        "step": step,
        "adapter_leaf_count": adapter_leaf_count,
        "resumable": include_train_state,
        **(metadata or {}),
    }
    (saved_step_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    if directory.exists():
        directory.rmtree()
    shutil.move(str(saved_step_dir), str(directory))
    tmp_directory.rmtree()


def prune_adapter_history(history_dir: epath.Path | str, keep: int) -> None:
    ################################################################################
    # DreamZero Eval Comparison: rolling compact adapter history
    ################################################################################
    # Keeps only the most recent compact resumable adapter checkpoints, so a run can
    # save frequently without accumulating full pi0.5 base parameters on scratch.
    ################################################################################
    if keep <= 0:
        return

    history_dir = epath.Path(history_dir).resolve()
    if not history_dir.exists():
        return

    step_dirs: list[tuple[int, epath.Path]] = []
    for path in history_dir.iterdir():
        if not path.is_dir() or path.name.endswith(".tmp") or not path.name.startswith("step_"):
            continue
        try:
            step = int(path.name.removeprefix("step_"))
        except ValueError:
            logging.warning("Ignoring adapter history directory with non-step name: %s", path)
            continue
        step_dirs.append((step, path))

    for _, path in sorted(step_dirs)[:-keep]:
        logging.info("Pruning old adapter history checkpoint: %s", path)
        path.rmtree()


def restore_adapter_state(
    directory: epath.Path | str,
    state: training_utils.TrainState,
) -> training_utils.TrainState:
    """Restore a minimal adapter_latest checkpoint into a base-initialized train state."""
    directory = epath.Path(directory).resolve()
    params_dir = directory / "params"
    train_state_dir = directory / "train_state"
    if not params_dir.exists():
        raise FileNotFoundError(f"Missing adapter params directory: {params_dir}")
    if not train_state_dir.exists():
        raise FileNotFoundError(f"Missing adapter train_state directory: {train_state_dir}")

    adapter_params = _model.restore_params(params_dir, restore_type=np.ndarray)
    full_params = _merge_adapter_params(adapter_params, state.params.to_pure_dict())

    new_params = state.params
    new_params.replace_by_pure_dict(full_params)

    with at.disable_typechecking():
        template = dataclasses.replace(state, params={}, ema_params=None)
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored_train_state = checkpointer.restore(train_state_dir, ocp.args.PyTreeRestore(item=template))

    return dataclasses.replace(
        restored_train_state,
        params=new_params,
        model_def=state.model_def,
        tx=state.tx,
        ema_decay=state.ema_decay,
        ema_params=None,
    )


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])


def _merge_adapter_params(adapter_params: at.Params, params: at.Params) -> at.Params:
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_adapter = flax.traverse_util.flatten_dict(adapter_params, sep="/")

    result = dict(flat_ref)
    missing = []
    mismatched = []
    for key, adapter_value in flat_adapter.items():
        ref_value = flat_ref.get(key)
        if ref_value is None:
            missing.append(key)
            continue
        if adapter_value.shape != ref_value.shape:
            mismatched.append((key, adapter_value.shape, ref_value.shape))
            continue
        result[key] = adapter_value.astype(ref_value.dtype) if adapter_value.dtype != ref_value.dtype else adapter_value

    if missing or mismatched:
        details = []
        if missing:
            details.append(f"{len(missing)} missing keys")
        if mismatched:
            details.append(f"{len(mismatched)} shape mismatches")
        raise ValueError(f"Adapter checkpoint is not compatible with the initialized params: {', '.join(details)}")

    logging.info("Restored %d adapter parameter leaves.", len(flat_adapter))
    return flax.traverse_util.unflatten_dict(result, sep="/")
