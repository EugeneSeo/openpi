import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class ShapeAwareCheckpointWeightLoader(WeightLoader):
    """Loads only checkpoint leaves whose shape matches the initialized model.

    ###########################################################################
    # DreamZero eval comparison note
    ###########################################################################
    # The Franka/ORCA pi0.5 comparison config changes pi0.5-base from its
    # default action dimension to 48D. Projection leaves tied to action/state
    # dimensionality therefore have incompatible shapes. This loader keeps all
    # matching pi0.5-base weights and leaves mismatched leaves randomly
    # initialized by returning the reference shape/dtype placeholders for them.
    ###########################################################################
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params_by_shape(loaded_params, params)


@dataclasses.dataclass(frozen=True)
class ShapeAwareAdapterWeightLoader(WeightLoader):
    """Loads pi0.5-base by shape, then overlays a compact adapter checkpoint.

    ###########################################################################
    # DreamZero eval comparison note
    ###########################################################################
    # Adapter checkpoints produced by TrainConfig.save_adapter_checkpoints store
    # only the trainable LoRA/action projection subset. To recover an inference
    # parameter tree, first reconstruct the 48D pi0.5 base with
    # ShapeAwareCheckpointWeightLoader, then overlay the adapter leaves.
    ###########################################################################
    """

    base_params_path: str
    adapter_params_path: str

    def load(self, params: at.Params) -> at.Params:
        base_params = ShapeAwareCheckpointWeightLoader(self.base_params_path).load(params)
        adapter_params = _model.restore_params(download.maybe_download(self.adapter_params_path), restore_type=np.ndarray)
        return _merge_adapter_params(adapter_params, base_params)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _merge_params_by_shape(loaded_params: at.Params, params: at.Params) -> at.Params:
    """Merge checkpoint leaves only when their shape matches the reference tree."""
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    result = {}
    skipped = []
    missing = []

    for k, ref_v in flat_ref.items():
        loaded_v = flat_loaded.get(k)
        if loaded_v is None:
            result[k] = ref_v
            missing.append(k)
            continue
        if loaded_v.shape != ref_v.shape:
            result[k] = ref_v
            skipped.append((k, loaded_v.shape, ref_v.shape))
            continue
        result[k] = loaded_v.astype(ref_v.dtype) if loaded_v.dtype != ref_v.dtype else loaded_v

    if skipped:
        logger.info("Skipped %d checkpoint leaves with shape mismatches.", len(skipped))
        for key, loaded_shape, ref_shape in skipped[:20]:
            logger.info("  %s: checkpoint %s -> model %s", key, loaded_shape, ref_shape)
        if len(skipped) > 20:
            logger.info("  ... %d more mismatched leaves omitted", len(skipped) - 20)
    if missing:
        logger.info("Filled %d missing checkpoint leaves from initialized model placeholders.", len(missing))

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _merge_adapter_params(adapter_params: at.Params, params: at.Params) -> at.Params:
    """Overlay adapter leaves onto a complete reference tree."""
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_adapter = flax.traverse_util.flatten_dict(adapter_params, sep="/")

    missing = []
    mismatched = []
    result = dict(flat_ref)

    for k, adapter_v in flat_adapter.items():
        ref_v = flat_ref.get(k)
        if ref_v is None:
            missing.append(k)
            continue
        if adapter_v.shape != ref_v.shape:
            mismatched.append((k, adapter_v.shape, ref_v.shape))
            continue
        result[k] = adapter_v.astype(ref_v.dtype) if adapter_v.dtype != ref_v.dtype else adapter_v

    if missing or mismatched:
        details = []
        if missing:
            details.append(f"{len(missing)} missing keys")
        if mismatched:
            details.append(f"{len(mismatched)} shape mismatches")
        raise ValueError(f"Adapter checkpoint is not compatible with the reference params: {', '.join(details)}")

    logger.info("Loaded %d adapter checkpoint leaves.", len(flat_adapter))
    return flax.traverse_util.unflatten_dict(result, sep="/")
