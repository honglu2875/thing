import multiprocessing

import jax.numpy as jnp
import numpy as np
import torch

import thing

_lock = multiprocessing.Lock()


def test_transmission():
    global _lock
    _lock.acquire()

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

    with thing.Server() as server:
        for obj in objects:
            thing.catch(obj)
            arr = server.get_array()
            assert (arr == obj).all()

    _lock.release()


def test_chunks():
    global _lock
    _lock.acquire()

    arr = np.random.rand(1000, 1000)  # default is float64
    with thing.Server() as server:
        thing.catch(arr)
        res = server.get_array()
        assert (arr == res).all()

    _lock.release()
