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
    client = thing.Client(server_addr="localhost", server_port="2876")
    with Servicer(port=2876) as server:
        for obj in objects:
            client.catch(obj, server="localhost:2876").wait()
            arr = server.get_tensor().data
            assert (arr == obj).all()


def test_chunks():
    arr = np.random.rand(1000, 1000)  # default is float64
    client = thing.Client(server_addr="localhost", server_port="2877")
    with Servicer(port=2877) as server:
        client.catch(arr, server="localhost:2877").wait()
        res = server.get_tensor().data
        assert (arr == res).all()


def test_strings():
    test = "slkdafjölkdsjölkfjdsljfa"
    client = thing.Client(server_addr="localhost", server_port="2878")
    with Servicer(port=2878) as server:
        client.catch(test, server="localhost:2878").wait()
        res = server.get_string().data
        assert test == res
