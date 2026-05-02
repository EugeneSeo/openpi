"""Minimal raw websocket client for an SO101 OpenPI policy server.

This intentionally avoids importing openpi_client so it can run from a small
local environment with only numpy, msgpack, and websockets installed.
"""

import argparse
import dataclasses
import functools
import logging
import time

import msgpack
import numpy as np
import websockets.sync.client


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    prompt: str = "pick up the object"
    num_requests: int = 1


def _pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


_packb = functools.partial(msgpack.packb, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


def _dummy_so101_observation(prompt: str) -> dict:
    return {
        "observation/images/front": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images/wrist": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros((6,), dtype=np.float32),
        "prompt": prompt,
    }


def main(args: Args) -> None:
    uri = f"ws://{args.host}:{args.port}"
    logging.info("Connecting to %s", uri)

    with websockets.sync.client.connect(uri, compression=None, max_size=None) as websocket:
        metadata = _unpackb(websocket.recv())
        logging.info("Server metadata: %s", metadata)

        for request_idx in range(args.num_requests):
            start = time.perf_counter()
            websocket.send(_packb(_dummy_so101_observation(args.prompt)))
            response = websocket.recv()
            elapsed_ms = (time.perf_counter() - start) * 1000

            if isinstance(response, str):
                raise RuntimeError(f"Server returned an error:\n{response}")

            result = _unpackb(response)
            actions = np.asarray(result["actions"])
            print(f"request={request_idx} actions.shape={actions.shape} elapsed_ms={elapsed_ms:.1f}")
            print(f"first_action={np.array2string(actions[0], precision=4, suppress_small=True)}")

            if actions.shape != (24, 6):
                raise RuntimeError(f"Expected actions shape (24, 6), got {actions.shape}")


def _parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=Args.host)
    parser.add_argument("--port", type=int, default=Args.port)
    parser.add_argument("--prompt", default=Args.prompt)
    parser.add_argument("--num-requests", type=int, default=Args.num_requests)
    return Args(**vars(parser.parse_args()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(_parse_args())
