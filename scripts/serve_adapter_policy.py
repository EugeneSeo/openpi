#!/usr/bin/env python3
"""Serve a compact OpenPI adapter checkpoint over websocket.

This is intentionally separate from serve_policy.py: it reconstructs a full
parameter tree from a base checkpoint plus an adapter_latest directory, then
uses the standard Policy/WebsocketPolicyServer stack for inference.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import socket

import flax.nnx as nnx
import jax
import tyro

from openpi import transforms
from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import weight_loaders as _weight_loaders


@dataclasses.dataclass
class Args:
    """Arguments for serving a compact adapter checkpoint."""

    # Training config name, e.g. pi05_so101_teleop_test_filtered.
    config: str = "pi05_so101_teleop_test_filtered"

    # Adapter checkpoint directory, e.g. .../<exp>/adapter_latest.
    adapter_dir: pathlib.Path = pathlib.Path(
        "/cluster/scratch/dohkim/openpi_checkpoints/pi05_so101_teleop_test_filtered/"
        "so101_lora_all_episodes_bs16_8k_65239922/adapter_latest"
    )

    # Base parameter checkpoint that the compact adapter overlays.
    base_params: str = "gs://openpi-assets/checkpoints/pi05_base/params"

    # If provided, injected when the observation does not include a prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000

    # Record the policy's behavior for debugging.
    record: bool = False


def _reference_params(train_config: _config.TrainConfig):
    """Create the model parameter tree used as the merge/check template."""
    model = nnx.eval_shape(train_config.model.create, jax.random.key(0))
    _, state = nnx.split(model)
    return state.to_pure_dict()


def create_adapter_policy(args: Args) -> _policy.Policy:
    """Create a Policy from a base checkpoint plus compact adapter params."""
    train_config = _config.get_config(args.config)
    adapter_dir = args.adapter_dir.resolve()
    adapter_params_dir = adapter_dir / "params"
    adapter_assets_dir = adapter_dir / "assets"

    if not adapter_params_dir.exists():
        raise FileNotFoundError(f"Missing adapter params directory: {adapter_params_dir}")
    if not adapter_assets_dir.exists():
        raise FileNotFoundError(f"Missing adapter assets directory: {adapter_assets_dir}")

    logging.info("Loading base params from: %s", args.base_params)
    logging.info("Overlaying adapter params from: %s", adapter_params_dir)
    loader = _weight_loaders.ShapeAwareAdapterWeightLoader(
        base_params_path=args.base_params,
        adapter_params_path=str(adapter_params_dir),
    )
    full_params = loader.load(_reference_params(train_config))
    model = train_config.model.load(full_params)

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load adapter norm stats.")
    norm_stats = _checkpoints.load_norm_stats(adapter_assets_dir, data_config.asset_id)

    return _policy.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(args.default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=train_config.policy_metadata,
    )


def main(args: Args) -> None:
    policy = create_adapter_policy(args)
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating adapter server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
