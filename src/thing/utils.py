import logging
import sys

import numpy as np

from thing import thing_pb2


def _get_default_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)-8s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


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
