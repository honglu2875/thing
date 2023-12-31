import time

import jax.numpy as jnp
import numpy as np
import pytest
import torch

import thing
from thing.servicer import Servicer
from thing.utils import _prepare_pytree_obj, _reconstruct_pytree_obj


def _equal(obj, obj2):
    """A hacky util function just for this test."""
    if isinstance(obj, (int, float, str)):
        return obj == obj2
    elif isinstance(obj, (list, tuple)):
        if len(obj) != len(obj2):
            return False
        for i in range(len(obj)):
            if not _equal(obj[i], obj2[i]):
                return False
        return True
    elif isinstance(obj, dict):
        if len(obj) != len(obj2):
            return False
        for k, v in obj.items():
            if k not in obj2:
                return False
            if not _equal(v, obj2[k]):
                return False
        return True
    elif isinstance(obj, (np.ndarray, torch.Tensor, jnp.ndarray)):
        return np.allclose(np.array(obj), np.array(obj2))
    elif obj is None:
        return obj2 is None
    else:
        raise TypeError(f"Unsupported type {type(obj)}")


@pytest.mark.parametrize(
    ("obj", "i"),
    (
        (
            (
                "a",
                1,
            ),
            0,
        ),
        ([np.array([1]), "b"], 1),
        ([["a"]], 2),
        ({"a": {"b": "c"}}, 3),
        (
            [
                None,
                1,
                "hello",
                [1, 2, 3],
                (1, 2, 3),
                {"a": 1, "b": 2},
                {"a": [1, 2, 3], "b": (1, 2, 3)},
                {"a": {"b": [1, 2, 3]}},
                np.array([1, 2, 3]),
                torch.tensor([1, 2, 3]),
                jnp.array([1, 2, 3]),
            ],
            4,
        ),
    ),
)
def test_pytree_obj_and_transmission(obj, i):
    root, id_to_leaves = _prepare_pytree_obj(obj)
    obj2 = _reconstruct_pytree_obj(root, id_to_leaves)

    assert _equal(obj, obj2)
    client = thing.Client(server_addr="localhost", server_port=2879 + i)
    with thing.Server(port=2879 + i) as server:
        client.catch(obj, name="a", server=f"localhost:{2879 + i}").result()
        res = server.store.get_object_by_name("a")
        assert _equal(res, obj)
