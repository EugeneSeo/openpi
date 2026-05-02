import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


SO101_ACTION_DIM = 6


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        front_image = _parse_image(data["observation/images/front"])
        wrist_image = _parse_image(data["observation/images/wrist"])

        image = {
            "base_0_rgb": front_image,
            "left_wrist_0_rgb": wrist_image,
            "right_wrist_0_rgb": np.zeros_like(front_image),
        }
        image_mask = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,
        }

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
class SO101Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :SO101_ACTION_DIM])}
