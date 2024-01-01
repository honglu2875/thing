import jax.numpy as jnp
import numpy as np
import pytest
import torch

from thing.utils import _get_size


@pytest.mark.parametrize(
    "obj",
    (
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        np.array([[1, 2], [3, 4]], dtype=np.int32),
        jnp.array([[1, 2], [3, 4]], dtype=np.int32),
    ),
)
def test_get_size(obj):
    s = _get_size(obj)
    assert s == 16
