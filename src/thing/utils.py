# Copyright 2023 Honglu Fan (https://github.com/honglu2875).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import re
import sys
from typing import Optional

import numpy as np

from thing import thing_pb2
from thing.type import TensorObject


def _set_up_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)-8s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


_numpy_dtypes = {
    "int8": thing_pb2.DTYPE.INT8,
    "int16": thing_pb2.DTYPE.INT16,
    "int32": thing_pb2.DTYPE.INT32,
    "int64": thing_pb2.DTYPE.INT64,
    "uint8": thing_pb2.DTYPE.UINT8,
    "uint16": thing_pb2.DTYPE.UINT16,
    "uint32": thing_pb2.DTYPE.UINT32,
    "uint64": thing_pb2.DTYPE.UINT64,
    "float16": thing_pb2.DTYPE.FLOAT16,
    "float32": thing_pb2.DTYPE.FLOAT32,
    "float64": thing_pb2.DTYPE.FLOAT64,
    "bool": thing_pb2.DTYPE.BOOL,
}

_to_numpy_dtypes = {
    thing_pb2.DTYPE.INT8: np.int8,
    thing_pb2.DTYPE.INT16: np.int16,
    thing_pb2.DTYPE.INT32: np.int32,
    thing_pb2.DTYPE.INT64: np.int64,
    thing_pb2.DTYPE.UINT8: np.uint8,
    thing_pb2.DTYPE.UINT16: np.uint16,
    thing_pb2.DTYPE.UINT32: np.uint32,
    thing_pb2.DTYPE.UINT64: np.uint64,
    thing_pb2.DTYPE.FLOAT16: np.float16,
    thing_pb2.DTYPE.FLOAT32: np.float32,
    thing_pb2.DTYPE.FLOAT64: np.float64,
    thing_pb2.DTYPE.BOOL: bool,
}


def _validate_server_name(
    server: str, logger: Optional[logging.Logger] = None
) -> str | None:
    if not server:
        return None

    pattern = re.compile(r"^(?P<host>[^:]+)(:(?P<port>\d+))?$")
    match = pattern.match(server)
    if not match:
        if logger:
            logger.warning(f"Invalid server address: {server}. Reverting to default.")
        return None
    return server


def reconstruct_array(chunks: list[thing_pb2.CatchArrayRequest]):
    chunks = sorted(chunks, key=lambda x: x.chunk_id)

    # TODO: is there a way to avoid copying the data?
    data = b"".join([chunk.data for chunk in chunks])
    arr = np.frombuffer(data, dtype=_to_numpy_dtypes[chunks[0].dtype]).reshape(
        chunks[0].shape
    )

    if chunks[0].framework == thing_pb2.FRAMEWORK.JAX:
        import jax.numpy as jnp

        arr = jnp.array(arr)
    elif chunks[0].framework == thing_pb2.FRAMEWORK.TORCH:
        import torch

        arr = torch.tensor(arr)

    return arr


def reconstruct_tensor_object(
    chunks: list[thing_pb2.CatchArrayRequest],
    client_addr: str = "unknown",
    timestamp: int = 0,
):
    array = reconstruct_array(chunks)
    return TensorObject.from_proto(
        chunks[0], array, client_addr=client_addr, timestamp=timestamp
    )
