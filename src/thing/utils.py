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
import ctypes
import dataclasses
import functools
import logging
import re
import secrets
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from numbers import Number
from typing import Any, Callable, Optional

import numpy as np

from thing import thing_pb2
from thing.type import (
    ArrayLike,
    Leaf,
    Object,
    PyTree,
    PyTreeObject,
    StringObject,
    TensorObject,
)
from thing.array_types import ByteWithMetadata, _class_str_to_framework


_used_hash = set()


def retry(num: int, delay: float):
    def _wrapper(func):
        @functools.wraps(func)
        def _retry(*args, **kwargs):
            last_error = RuntimeError()
            for _ in range(num):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    time.sleep(delay)
            raise last_error

        return _retry

    return _wrapper


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


def reconstruct_tensor_object(
    chunks: list[thing_pb2.Array],
    client_addr: str = "unknown",
    timestamp: int = 0,
) -> TensorObject:
    array = ByteWithMetadata.from_proto(chunks).get_array()
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
    """Whether an object is a numpy/torch/jax tensor."""
    # This ugly function is to avoid importing all three frameworks.
    # Note that we aim to be as non-intrusive as possible.
    # However, robustness is taking a big hit, so be aware.
    if isinstance(obj, Number) or str(obj.__class__) in _class_str_to_framework:
        return True
    # We don't support subclasses of the three major tensor classes.
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


def _is_small(obj: Leaf) -> bool:
    """Deciding whether the leaf object is small enough so that we do not create
    a separate request."""
    if isinstance(obj, str) and sys.getsizeof(obj) <= 256:
        return True
    if _is_tensor(obj) and ByteWithMetadata._get_size(obj) <= 256:
        return True
    return False


def _prepare_string(obj, idx=None, name=None) -> thing_pb2.String:
    return thing_pb2.String(
        id=idx or get_rand_id(),
        var_name=name,
        data=obj,
    )


def _prepare_array(
    obj: Any,
    idx: Optional[int] = None,
    name: Optional[str] = None,
    shape: Optional[tuple] = None,
    dtype: Optional[Any] = None,
    framework: Optional[Any] = None,
    chunk_id: Optional[int] = None,
    num_chunks: Optional[int] = None,
) -> thing_pb2.Array:
    if isinstance(obj, bytes):
        # obj is already ready to be sent
        if not all(p is not None for p in [shape, dtype, framework]):
            raise ValueError(
                "When the input are bytes, all of shape, dtype, framework must be given."
            )
        data = obj
    else:
        # shape, dtype, framework need to be determined,
        # and obj needs to be converted to bytes
        byte_obj = ByteWithMetadata.from_array(obj)
        framework = byte_obj.framework
        data = byte_obj.data
        shape = byte_obj.shape
        dtype = byte_obj.dtype
    return thing_pb2.Array(
        id=idx or get_rand_id(),
        var_name=name,
        shape=shape,
        dtype=dtype,
        framework=framework,
        data=data,
        chunk_id=chunk_id,
        num_chunks=num_chunks,
    )


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
        string, array = None, None
        if _is_small(obj):
            if isinstance(obj, str):
                string = _prepare_string(obj, idx=idx)
            else:
                array = _prepare_array(obj, idx=idx)
        else:
            id_to_leaves[idx] = obj
        root = thing_pb2.PyTreeNode(
            id=idx,
            var_name=name,
            node_type=_get_node_type(obj),
            children=[],
            object_id=idx,
            key=key,
            string=string,
            array=array,
        )
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
        # If data is attached, directly restore them
        if root.node_type == thing_pb2.NODE_TYPE.STRING and root.HasField("string"):
            return root.string.data
        elif root.node_type == thing_pb2.NODE_TYPE.TENSOR and root.HasField("array"):
            return reconstruct_tensor_object([root.array]).data
        # Otherwise, the object must have been sent separately and is identifiable
        if root.object_id not in id_to_leaves:
            raise NameError
        # Look up the object and unravel the content
        leaf = id_to_leaves[root.object_id]
        if isinstance(leaf, Object):
            leaf = leaf.data
        return leaf
    else:
        raise TypeError(f"Unsupported type {root.node_type}")
