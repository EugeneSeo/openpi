from __future__ import annotations

import dataclasses
import functools
import json
import logging
import os
import pathlib
import platform

from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.weight_loaders as _weight_loaders
import train as train_script


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str = "pi05_franka_orca_bag_groceries"
    checkpoint_dir: str = ""
    history_subdir: str = "adapter_history"
    latest_n: int = 10
    val_batches: int = 2
    batch_size: int | None = None
    num_workers: int | None = None
    split: str = "val"
    split_path: str | None = None
    include_adapter_latest: bool = False
    output_json: str | None = None
    output_md: str | None = None


def _collect_checkpoint_dirs(args: Args) -> list[pathlib.Path]:
    checkpoint_dir = pathlib.Path(args.checkpoint_dir).resolve()
    history_root = checkpoint_dir / args.history_subdir
    if not history_root.exists():
        raise FileNotFoundError(f"Missing adapter history directory: {history_root}")

    def _step_key(path: pathlib.Path) -> int:
        return int(path.name.removeprefix("step_"))

    dirs = sorted(
        [p for p in history_root.iterdir() if p.is_dir() and p.name.startswith("step_")],
        key=_step_key,
    )
    if args.latest_n > 0:
        dirs = dirs[-args.latest_n :]

    if args.include_adapter_latest:
        latest_dir = checkpoint_dir / "adapter_latest"
        if latest_dir.exists():
            dirs.append(latest_dir)

    if not dirs:
        raise ValueError(f"No adapter checkpoints found under {history_root}")
    return dirs


def _load_metadata(directory: pathlib.Path) -> dict[str, object]:
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    return {}


def _write_outputs(results: list[dict[str, object]], output_json: pathlib.Path, output_md: pathlib.Path) -> None:
    output_json.write_text(json.dumps(results, indent=2) + "\n")

    lines = [
        "# Adapter History Validation Loss",
        "",
        "| checkpoint | step | val_loss |",
        "| --- | ---: | ---: |",
    ]
    for row in results:
        lines.append(f"| {row['name']} | {row['step']} | {row['val_loss']:.6f} |")

    best = min(results, key=lambda row: float(row["val_loss"]))
    lines.extend(
        [
            "",
            f"Best checkpoint: `{best['name']}` at step `{best['step']}` with val loss `{best['val_loss']:.6f}`.",
            "",
        ]
    )
    output_md.write_text("\n".join(lines))


def main(args: Args) -> None:
    train_script.init_logging()
    logging.info("Running on: %s", platform.node())

    checkpoint_dirs = _collect_checkpoint_dirs(args)
    checkpoint_root = pathlib.Path(args.checkpoint_dir).resolve()

    config = _config.get_config(args.config_name)
    if not dataclasses.is_dataclass(config.data) or not hasattr(config.data, "split"):
        raise ValueError("This eval script requires a data config with a 'split' field.")

    data_kwargs = {"split": args.split}
    if args.split_path is not None:
        data_kwargs["split_path"] = args.split_path
    data_config = dataclasses.replace(config.data, **data_kwargs)

    config = dataclasses.replace(
        config,
        exp_name=f"eval_{checkpoint_root.name}",
        data=data_config,
        batch_size=args.batch_size if args.batch_size is not None else config.batch_size,
        num_workers=args.num_workers if args.num_workers is not None else config.num_workers,
        wandb_enabled=False,
        log_wandb_images=False,
    )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax_cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "~/.cache/jax")
    jax.config.update("jax_compilation_cache_dir", str(pathlib.Path(jax_cache_dir).expanduser()))

    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    val_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=False,
        num_batches=args.val_batches,
    )

    base_state, train_state_sharding = train_script.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(base_state)
    base_params = base_state.params.to_pure_dict()

    pval_step = jax.jit(
        functools.partial(train_script.val_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    results: list[dict[str, object]] = []
    for directory in checkpoint_dirs:
        metadata = _load_metadata(directory)
        adapter_params = _model.restore_params(directory / "params", restore_type=np.ndarray)
        full_params = _weight_loaders._merge_adapter_params(adapter_params, base_params)
        base_state.params.replace_by_pure_dict(full_params)

        val_infos = []
        val_iter = iter(val_loader)
        for batch_idx, val_batch in zip(range(args.val_batches), val_iter, strict=False):
            val_rng = jax.random.fold_in(rng, batch_idx)
            with sharding.set_mesh(mesh):
                val_infos.append(pval_step(val_rng, base_state, val_batch))

        stacked_val_infos = common_utils.stack_forest(val_infos)
        reduced_val_info = jax.device_get(jax.tree.map(jnp.mean, stacked_val_infos))
        val_loss = float(reduced_val_info["val_loss"])

        step = int(metadata.get("step", directory.name.removeprefix("step_") or 0))
        row = {
            "name": directory.name,
            "path": str(directory),
            "step": step,
            "val_loss": val_loss,
        }
        results.append(row)
        logging.info("Evaluated %s: step=%d val_loss=%.6f", directory.name, step, val_loss)

    results.sort(key=lambda row: int(row["step"]))
    output_json = pathlib.Path(args.output_json) if args.output_json else checkpoint_root / "adapter_history_val_eval.json"
    output_md = pathlib.Path(args.output_md) if args.output_md else checkpoint_root / "adapter_history_val_eval.md"
    _write_outputs(results, output_json, output_md)

    best = min(results, key=lambda row: float(row["val_loss"]))
    logging.info("Best checkpoint: %s (step=%s, val_loss=%.6f)", best["name"], best["step"], best["val_loss"])
    print(json.dumps({"best": best, "output_json": str(output_json), "output_md": str(output_md)}, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
