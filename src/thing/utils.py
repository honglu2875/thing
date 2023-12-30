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
import secrets
import sys
from numbers import Number
from typing import Optional

import numpy as np

from thing import thing_pb2
from thing.type import (ArrayLike, Object, PyTree, PyTreeObject, StringObject,
                        TensorObject)

_used_hash = set()


def get_rand_id():
    global _used_hash
    while (idx := secrets.randbelow(2**63 - 1)) in _used_hash:
        pass
    _used_hash.add(idx)
    return idx


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


def reconstruct_array(chunks: list[thing_pb2.Array]) -> ArrayLike:
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
    chunks: list[thing_pb2.Array],
    client_addr: str = "unknown",
    timestamp: int = 0,
) -> TensorObject:
    array = reconstruct_array(chunks)
    return TensorObject.from_proto(
        chunks[0], array, client_addr=client_addr, timestamp=timestamp
    )


def reconstruct_string_object(
    string_request: thing_pb2.String,
    client_addr: str = "unknown",
    timestamp: int = 0,
) -> StringObject:
    data = string_request.data
    return StringObject.from_proto(
        string_request, data, client_addr=client_addr, timestamp=timestamp
    )


def reconstruct_pytree_object(
    pytree_request: thing_pb2.PyTreeNode,
    client_addr: str = "unknown",
    timestamp: int = 0,
) -> PyTreeObject:
    data = pytree_request  # Unravel the pytree object at a later time
    return PyTreeObject.from_proto(
        pytree_request, data, client_addr=client_addr, timestamp=timestamp
    )


def _is_tensor(obj) -> bool:
    # Avoid importing all three major frameworks!!
    # Robustness is taking a hit, so be aware.
    if (
        isinstance(obj, Number)
        or str(obj.__class__)
        in [
            "<class 'numpy.ndarray'>",
            "<class 'torch.Tensor'>",
        ]
        or "ArrayImpl" in str(obj.__class__)
    ):
        return True
    return False


def _get_node_type(obj) -> int:
    if isinstance(obj, tuple):
        return thing_pb2.NODE_TYPE.TUPLE
    elif isinstance(obj, list):
        return thing_pb2.NODE_TYPE.LIST
    elif isinstance(obj, dict):
        return thing_pb2.NODE_TYPE.DICT
    elif isinstance(obj, str):
        return thing_pb2.NODE_TYPE.STRING
    elif _is_tensor(obj):
        return thing_pb2.NODE_TYPE.TENSOR
    else:
        raise TypeError(f"Unsupported type {type(obj)}")


def _prepare_pytree_obj(obj, id_to_leaves=None, key=None, name=None) -> tuple:
    id_to_leaves = {} if id_to_leaves is None else id_to_leaves
    idx = get_rand_id()

    if obj is None:
        root = thing_pb2.PyTreeNode(
            id=idx,
            var_name=name,
            node_type=thing_pb2.NODE_TYPE.NONE,
            children=[],
            key=key,
        )
    elif isinstance(obj, (tuple, list)):
        root = thing_pb2.PyTreeNode(
            id=idx,
            var_name=name,
            node_type=_get_node_type(obj),
            children=[_prepare_pytree_obj(child, id_to_leaves)[0] for child in obj],
            key=key,
        )
    elif isinstance(obj, dict):
        root = thing_pb2.PyTreeNode(
            id=idx,
            var_name=name,
            node_type=_get_node_type(obj),
            children=[
                _prepare_pytree_obj(child, id_to_leaves, key=k)[0]
                for k, child in obj.items()
            ],
            key=key,
        )
    elif _is_tensor(obj) or isinstance(obj, str):
        root = thing_pb2.PyTreeNode(
            id=idx,
            var_name=name,
            node_type=_get_node_type(obj),
            children=[],
            object_id=idx,
            key=key,
        )
        id_to_leaves[idx] = obj
    else:
        raise TypeError(f"Unsupported type {type(obj)}")

    return root, id_to_leaves


def _reconstruct_pytree_obj(
    root: thing_pb2.PyTreeNode, id_to_leaves: dict
) -> Optional[PyTree]:
    if root.node_type == thing_pb2.NODE_TYPE.NONE:
        return None
    elif root.node_type == thing_pb2.NODE_TYPE.TUPLE:
        return tuple(
            [_reconstruct_pytree_obj(child, id_to_leaves) for child in root.children]
        )
    elif root.node_type == thing_pb2.NODE_TYPE.LIST:
        return [_reconstruct_pytree_obj(child, id_to_leaves) for child in root.children]
    elif root.node_type == thing_pb2.NODE_TYPE.DICT:
        return {
            child.key: _reconstruct_pytree_obj(child, id_to_leaves)
            for child in root.children
        }
    elif root.node_type in [thing_pb2.NODE_TYPE.TENSOR, thing_pb2.NODE_TYPE.STRING]:
        if root.object_id not in id_to_leaves:
            raise NameError
        leaf = id_to_leaves[root.object_id]
        if isinstance(leaf, Object):  # unravel wrapped objects
            leaf = leaf.data
        return leaf
    else:
        raise TypeError(f"Unsupported type {root.node_type}")
