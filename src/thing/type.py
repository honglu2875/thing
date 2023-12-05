import dataclasses
import time
from enum import Enum
from typing import Any, Optional

import thing.thing_pb2


class FRAMEWORK(Enum):
    NUMPY = 0
    TORCH = 1
    JAX = 2


@dataclasses.dataclass
class TensorObject:
    """
    The received tensor object with metadata.
    """

    id: int
    name: Optional[str]
    shape: tuple
    data: Any  # The reconstructed torch, jax, or numpy array
    timestamp: int
    client_addr: str

    @classmethod
    def from_proto(
        cls,
        proto: thing.thing_pb2.CatchArrayRequest,
        array: Any,
        client_addr: str,
        timestamp: Optional[int] = None,
    ):
        """
        Reconstruct a TensorObject from a protobuf (for the metadata) and a fully reconstructed array.

        Args:
            proto: the protobuf containing the metadata.
            array: the reconstructed array.
            client_addr: the address of the client that sent the array.
            timestamp: the timestamp of the array.
        Returns:
            The reconstructed TensorObject.
        """
        timestamp = int(time.time()) if timestamp is None else timestamp
        return cls(
            id=proto.id,
            name=proto.var_name,
            shape=tuple(proto.shape),
            data=array,
            timestamp=timestamp,
            client_addr=client_addr,
        )
