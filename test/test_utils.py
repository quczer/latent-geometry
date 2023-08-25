import math

import numpy as np
import pytest

from latent_geometry.utils import batchify


@pytest.mark.parametrize(
    "x,y,batch_size",
    [
        (
            np.arange(10 * 4 * 2).reshape(10, 4, 2),
            np.random.randn(10, 4, 2),
            1,
        ),
        (
            np.arange(10 * 4 * 2).reshape(10, 4, 2),
            np.random.randn(10, 4, 2),
            4,
        ),
        (
            np.arange(10 * 4 * 2).reshape(10, 4, 2),
            np.random.randn(10, 4, 2),
            10 * 4 * 2,
        ),
    ],
)
def test_batchify(x, y, batch_size):
    counter = 0
    B = x.shape[0]

    def foo(x: np.array, y: np.array):
        nonlocal counter
        counter += 1
        return x**2 + np.cos(y)

    batch_foo = batchify(batch_size=batch_size)(foo)

    res_foo = foo(x, y)
    res_batch_foo = batch_foo(x, y)

    assert np.allclose(res_foo, res_batch_foo)
    assert counter == math.ceil(B / (batch_size or B)) + 1
