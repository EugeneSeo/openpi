#!/usr/bin/env python3
"""Export a compact LoRA/action adapter checkpoint from a full OpenPI checkpoint."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import shutil

import flax.traverse_util
import numpy as np
import orbax.checkpoint as ocp
import tyro

import openpi.models.model as _model


_ADAPTER_PATTERNS = (
    re.compile(r".*lora.*"),
    re.compile(r".*action_in_proj.*"),
    re.compile(r".*action_out_proj.*"),
    re.compile(r".*time_mlp_in.*"),
    re.compile(r".*time_mlp_out.*"),
)


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint_dir: pathlib.Path
    output_dir: pathlib.Path
    overwrite: bool = False


def _is_adapter_leaf(path: str) -> bool:
    return any(pattern.fullmatch(path) for pattern in _ADAPTER_PATTERNS)


def _tree_nbytes(tree) -> int:
    total = 0
    for leaf in flax.traverse_util.flatten_dict(tree, sep="/").values():
        if hasattr(leaf, "nbytes"):
            total += int(leaf.nbytes)
        else:
            total += np.asarray(leaf).nbytes
    return total


def main(args: Args) -> None:
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    params_dir = output_dir / "params"

    if not (checkpoint_dir / "params").exists():
        raise FileNotFoundError(f"Missing params directory: {checkpoint_dir / 'params'}")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_params = _model.restore_params(checkpoint_dir / "params", restore_type=np.ndarray)
    flat_full = flax.traverse_util.flatten_dict(full_params, sep="/")
    flat_adapter = {path: value for path, value in flat_full.items() if _is_adapter_leaf(path)}
    if not flat_adapter:
        raise RuntimeError(f"No adapter leaves matched in {checkpoint_dir / 'params'}")

    adapter_params = flax.traverse_util.unflatten_dict(flat_adapter, sep="/")
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(params_dir, {"params": adapter_params})

    assets_src = checkpoint_dir / "assets"
    if assets_src.exists():
        shutil.copytree(assets_src, output_dir / "assets")

    source_metadata = {}
    metadata_path = checkpoint_dir / "metadata.json"
    if metadata_path.exists():
        source_metadata = json.loads(metadata_path.read_text())

    metadata = {
        "source_checkpoint": str(checkpoint_dir),
        "source_metadata": source_metadata,
        "adapter_leaf_count": len(flat_adapter),
        "adapter_nbytes": _tree_nbytes(adapter_params),
        "adapter_patterns": [pattern.pattern for pattern in _ADAPTER_PATTERNS],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    restored = _model.restore_params(params_dir, restore_type=np.ndarray)
    restored_count = len(flax.traverse_util.flatten_dict(restored, sep="/"))
    if restored_count != len(flat_adapter):
        raise RuntimeError(f"Restored {restored_count} leaves, expected {len(flat_adapter)}")

    print(f"Wrote adapter checkpoint: {output_dir}")
    print(f"Adapter leaves: {len(flat_adapter)}")
    print(f"Adapter size: {_tree_nbytes(adapter_params) / 1024**3:.3f} GiB")


if __name__ == "__main__":
    main(tyro.cli(Args))
