import dataclasses
import time
from concurrent.futures import Future
from enum import Enum
from typing import Any, Optional, Union

from thing import thing_pb2


class FRAMEWORK(Enum):
    NUMPY = 0
    TORCH = 1
    JAX = 2


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
            thing_pb2.CatchArrayRequest,
            thing_pb2.CatchStringRequest,
            thing_pb2.PyTreeNode,
        ],
        obj: Any,
        client_addr: str,
        timestamp: Optional[int] = None,
    ):
        """
        Reconstruct a Object from a protobuf (for the metadata) and a fully reconstructed array.

        Args:
            proto: the protobuf containing the metadata.
            obj: the reconstructed object.
            client_addr: the address of the client that sent the array.
            timestamp: the timestamp of the array.
        Returns:
            The reconstructed TensorObject.
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


class Futures:
    def __init__(self, futures: list[Future]):
        self._futures = futures

    def result(self, timeout=None):
        return [f.result(timeout=timeout) for f in self._futures]
