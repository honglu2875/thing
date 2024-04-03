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
from thing import thing_pb2
import dataclasses
from typing import Any
from thing.type import ArrayLike
from numbers import Number
import sys
import numpy as np
import ctypes


class ReversibleDict(dict):
    _reversed = None

    def get_key(self, v: Any, default=None):
        try:
            hash(v)
        except TypeError as e:
            raise ValueError(
                f"Input of get_key must be hashable, got {type(v)} instead."
            ) from e

        if self._reversed is None:
            self._reversed = {v: k for k, v in self.items()}

        return self._reversed.get(v, default)


_numpy_dtypes = ReversibleDict(
    {
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
)

_torch_dtypes = ReversibleDict(
    {
        "torch.int8": thing_pb2.DTYPE.INT8,
        "torch.int16": thing_pb2.DTYPE.INT16,
        "torch.int32": thing_pb2.DTYPE.INT32,
        "torch.int64": thing_pb2.DTYPE.INT64,
        "torch.uint8": thing_pb2.DTYPE.UINT8,
        # torch 2.1 does not support UINT16, UINT32, UINT64
        "torch.float16": thing_pb2.DTYPE.FLOAT16,
        "torch.bfloat16": thing_pb2.DTYPE.BFLOAT16,
        "torch.float32": thing_pb2.DTYPE.FLOAT32,
        "torch.float64": thing_pb2.DTYPE.FLOAT64,
        "torch.bool": thing_pb2.DTYPE.BOOL,
    }
)

_jax_dtypes = ReversibleDict(
    {
        "int8": thing_pb2.DTYPE.INT8,
        "int16": thing_pb2.DTYPE.INT16,
        "int32": thing_pb2.DTYPE.INT32,
        "int64": thing_pb2.DTYPE.INT64,
        "uint8": thing_pb2.DTYPE.UINT8,
        "uint16": thing_pb2.DTYPE.UINT16,
        "uint32": thing_pb2.DTYPE.UINT32,
        "uint64": thing_pb2.DTYPE.UINT64,
        "float16": thing_pb2.DTYPE.FLOAT16,
        "bfloat16": thing_pb2.DTYPE.BFLOAT16,
        "float32": thing_pb2.DTYPE.FLOAT32,
        "float64": thing_pb2.DTYPE.FLOAT64,
        "bool": thing_pb2.DTYPE.BOOL,
    }
)


# Sacrifice type-check robustness to avoid unnecessary imports
_class_str_to_framework = {
    # Numpy array naming should be stable enough
    "<class 'numpy.ndarray'>": thing_pb2.FRAMEWORK.NUMPY,
    # Torch array naming is also considered stable
    "<class 'torch.Tensor'>": thing_pb2.FRAMEWORK.TORCH,
    # However, I am a bit worried about JAX...
    "<class 'jaxlib.xla_extension.ArrayImpl'>": thing_pb2.FRAMEWORK.JAX,
}


@dataclasses.dataclass
class ByteWithMetadata:
    framework: thing_pb2.FRAMEWORK
    shape: tuple
    dtype: thing_pb2.DTYPE
    data: bytes

    @staticmethod
    def _get_framework(obj: ArrayLike) -> thing_pb2.FRAMEWORK:
        return _class_str_to_framework[str(obj.__class__)]

    @staticmethod
    def _get_address(obj: ArrayLike, framework: thing_pb2.FRAMEWORK) -> int:
        match framework:
            case thing_pb2.FRAMEWORK.NUMPY:
                return obj.ctypes.data
            case thing_pb2.FRAMEWORK.JAX:
                return np.array(obj, copy=False).ctypes.data
            case thing_pb2.FRAMEWORK.TORCH:
                return obj.data_ptr()

    @staticmethod
    def _get_shape(obj: ArrayLike, framework: thing_pb2.FRAMEWORK) -> tuple:
        return tuple(obj.shape)

    @staticmethod
    def _to_bytes_no_copy(address: int, nbytes: int) -> bytes:
        """Convert numpy array to bytes without copying."""
        # Use ctypes instead of `array.tobytes()` to avoid creating a copy
        return (ctypes.c_char * nbytes).from_address(address).raw

    @staticmethod
    def _get_size(obj: ArrayLike) -> int:
        """A framework-agnostic way of getting the total size of an array."""
        if isinstance(obj, Number):
            return sys.getsizeof(obj)
        return obj.nbytes

    @staticmethod
    def _get_proto_dtype(
        obj: ArrayLike, framework: thing_pb2.FRAMEWORK
    ) -> thing_pb2.DTYPE:
        match framework:
            case thing_pb2.FRAMEWORK.NUMPY:
                return _numpy_dtypes[obj.dtype.name]
            case thing_pb2.FRAMEWORK.TORCH:
                return _torch_dtypes[str(obj.dtype)]
            case thing_pb2.FRAMEWORK.JAX:
                return _jax_dtypes[obj.dtype.name]

    def get_dtype(self):
        match self.framework:
            case thing_pb2.FRAMEWORK.NUMPY:
                import numpy

                key = _numpy_dtypes.get_key(self.dtype)
                if key == "bool":
                    return bool
                else:
                    return eval("numpy." + key)
            case thing_pb2.FRAMEWORK.TORCH:
                import torch

                return eval(_torch_dtypes.get_key(self.dtype))
            case thing_pb2.FRAMEWORK.JAX:
                import jax

                return eval("jax.numpy." + _jax_dtypes.get_key(self.dtype))

    def get_array(self):
        # This method is used for the receivers. The implementations below
        # may copy the bytes but it is much less of a concern than disturbing
        # the server's memory footprint.

        match self.framework:
            case thing_pb2.FRAMEWORK.NUMPY:
                import numpy

                return numpy.frombuffer(self.data, dtype=self.get_dtype()).reshape(
                    self.shape
                )
            case thing_pb2.FRAMEWORK.TORCH:
                import torch

                return torch.frombuffer(
                    bytearray(self.data), dtype=self.get_dtype()
                ).view(*self.shape)
            case thing_pb2.FRAMEWORK.JAX:
                import jax

                return jax.numpy.frombuffer(self.data, dtype=self.get_dtype()).reshape(
                    self.shape
                )

    @classmethod
    def from_array(cls, array: ArrayLike):
        if isinstance(array, Number):
            array = np.array(array)
        framework = cls._get_framework(array)
        nbytes = cls._get_size(array)
        shape = cls._get_shape(array, framework)
        address = cls._get_address(array, framework)
        bytes = cls._to_bytes_no_copy(address, nbytes)
        dtype = cls._get_proto_dtype(array, framework)
        return cls(framework=framework, shape=shape, dtype=dtype, data=bytes)

    @classmethod
    def from_proto(cls, proto: thing_pb2.Array | list[thing_pb2.Array]):
        if isinstance(proto, list):
            # Reading from a list of chunks
            if not proto:
                raise ValueError("Input cannot be an empty list of chunks.")
            proto = sorted(proto, key=lambda x: x.chunk_id)
            # TODO: is there a way to avoid copying the data?
            data = b"".join([chunk.data for chunk in proto])
            return cls(
                framework=proto[0].framework,
                shape=proto[0].shape,
                dtype=proto[0].dtype,
                data=data,
            )
        return cls(
            framework=proto.framework,
            shape=proto.shape,
            dtype=proto.dtype,
            data=proto.bytes,
        )
