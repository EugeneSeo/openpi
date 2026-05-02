"""Send a dummy SO101 observation to a websocket policy server."""

import dataclasses
import logging
import time

import numpy as np
from openpi_client import websocket_client_policy
import tyro


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    prompt: str = "pick up the object"
    num_requests: int = 1


def _dummy_so101_observation(prompt: str) -> dict:
    return {
        "observation/images/front": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images/wrist": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros((6,), dtype=np.float32),
        "prompt": prompt,
    }


def main(args: Args) -> None:
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info("Server metadata: %s", policy.get_server_metadata())

    for request_idx in range(args.num_requests):
        start = time.perf_counter()
        result = policy.infer(_dummy_so101_observation(args.prompt))
        elapsed_ms = (time.perf_counter() - start) * 1000

        actions = np.asarray(result["actions"])
        print(f"request={request_idx} actions.shape={actions.shape} elapsed_ms={elapsed_ms:.1f}")
        print(f"first_action={np.array2string(actions[0], precision=4, suppress_small=True)}")

        if actions.shape != (24, 6):
            raise RuntimeError(f"Expected actions shape (24, 6), got {actions.shape}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
