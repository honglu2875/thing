import jax.numpy as jnp
import numpy as np
import pytest
import torch

from thing.array_types import ByteWithMetadata


@pytest.mark.parametrize(
    "obj",
    (
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        np.array([[1, 2], [3, 4]], dtype=np.int32),
        jnp.array([[1, 2], [3, 4]], dtype=np.int32),
    ),
)
def test_get_size(obj):
    s = ByteWithMetadata._get_size(obj)
    assert s == 16
