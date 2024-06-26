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
import dataclasses
import time
from concurrent.futures import Future
from enum import Enum
from numbers import Number
from typing import Any, Optional, Union

from thing import thing_pb2


ArrayLike = Union["np.ndarray", "torch.Tensor", "jaxlib.xla_extension.ArrayImpl"]
Leaf = Optional[Union[ArrayLike, str, Number]]
PyTree = Union[tuple, list, dict, Leaf]


@dataclasses.dataclass
class Object:
    """
    The received tensor object with metadata.
    """

    id: int
    name: Optional[str]
    data: Any  # The reconstructed torch, jax, numpy array, string, or pytree object
    timestamp: int
    client_addr: str
    __extra_fields__ = ()

    @classmethod
    def from_proto(
        cls,
        proto: Union[
            thing_pb2.Array,
            thing_pb2.String,
            thing_pb2.PyTreeNode,
        ],
        obj: Any,
        client_addr: str,
        timestamp: Optional[int] = None,
    ):
        """
        Reconstruct an Object from a protobuf (for the metadata) and a fully reconstructed array.

        Args:
            proto: the protobuf containing the metadata.
            obj: the reconstructed object.
            client_addr: the address of the client that sent the array.
            timestamp: the timestamp of the array.
        Returns:
            The reconstructed Object.
        """
        timestamp = int(time.time()) if timestamp is None else timestamp
        return cls(
            id=proto.id,
            name=proto.var_name,
            data=obj,
            timestamp=timestamp,
            client_addr=client_addr,
            **{k: getattr(proto, k) for k in cls.__extra_fields__},
        )


@dataclasses.dataclass
class TensorObject(Object):
    shape: tuple
    __extra_fields__ = ("shape",)


@dataclasses.dataclass
class StringObject(Object):
    ...


@dataclasses.dataclass
class PyTreeObject(Object):
    ...


class Awaitable:
    """A wrapper around Future object.

    Mainly for cosmetic: one can call `thing.catch(v).wait()` to wait for the result.
    """

    def __init__(self, future: Future):
        self._future = future

    def wait(self):
        return self._future.result()
