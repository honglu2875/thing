import jax.numpy as jnp
import numpy as np
import pytest
import torch

import thing
from thing.servicer import Servicer


def test_transmission():
    objects = []
    for dtype in [
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        bool,
    ]:
        objects.append(np.array([1, 2, 3], dtype=dtype))

    for dtype in [
        torch.float16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.bool,
    ]:
        objects.append(torch.tensor([1, 2, 3], dtype=dtype))

    for dtype in [
        jnp.float16,
        jnp.float32,
        jnp.int8,
        jnp.int16,
        jnp.int32,
        jnp.uint8,
        jnp.uint16,
        jnp.uint32,
        bool,
    ]:
        objects.append(jnp.array([1, 2, 3], dtype=dtype))

    with Servicer(port=2876) as server:
        for obj in objects:
            thing.catch(obj, server="localhost:2876")
            arr = server.get_tensor().data
            assert (arr == obj).all()


def test_chunks():
    arr = np.random.rand(1000, 1000)  # default is float64
    with Servicer(port=2877) as server:
        thing.catch(arr, server="localhost:2877")
        res = server.get_tensor().data
        assert (arr == res).all()


def test_strings():
    test = "slkdafjölkdsjölkfjdsljfa"
    with Servicer(port=2878) as server:
        thing.catch(test, server="localhost:2878")
        res = server.get_string().data
        assert test == res
