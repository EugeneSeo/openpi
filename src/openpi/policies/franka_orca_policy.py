import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


FRANKA_ORCA_ACTION_DIM = 48


def make_franka_orca_example() -> dict:
    """Creates a random input example for the Franka/ORCA policy."""
    return {
        "observation/oakd_front_view": np.random.randint(256, size=(540, 960, 3), dtype=np.uint8),
        "observation/aria_rgb_cam": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/state": np.random.rand(FRANKA_ORCA_ACTION_DIM).astype(np.float32),
        "prompt": "bag the groceries",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaOrcaInputs(transforms.DataTransformFn):
    """
    Adapter for the DreamZero Franka/ORCA bag-groceries comparison dataset.

    This is intentionally narrow: the dataset is bimanual Franka arms plus ORCA
    hands, represented as 48D state/action vectors in this order:
      left arm 0:7, left hand 7:24, right arm 24:31, right hand 31:48.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/oakd_front_view"])
        aria_image = _parse_image(data["observation/aria_rgb_cam"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                image = {
                    "base_0_rgb": base_image,
                    "left_wrist_0_rgb": aria_image,
                    "right_wrist_0_rgb": np.zeros_like(base_image),
                }
                image_mask = {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                    "right_wrist_0_rgb": np.False_,
                }
            case _model.ModelType.PI0_FAST:
                image = {
                    "base_0_rgb": base_image,
                    "base_1_rgb": np.zeros_like(base_image),
                    "wrist_0_rgb": aria_image,
                }
                image_mask = {
                    "base_0_rgb": np.True_,
                    "base_1_rgb": np.True_,
                    "wrist_0_rgb": np.True_,
                }
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": image,
            "image_mask": image_mask,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaOrcaOutputs(transforms.DataTransformFn):
    """Return the 48D Franka/ORCA action chunk expected by the downstream controller."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :FRANKA_ORCA_ACTION_DIM])}
