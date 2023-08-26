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
def test_batchify_on_function(x, y, batch_size):
    counter = 0
    B = x.shape[0]

    def foo(x: np.array, y: np.array):
        nonlocal counter
        counter += 1
        return x**2 + np.cos(y)

    @batchify(batch_size=batch_size)
    def bar(x: np.array, y: np.array):
        nonlocal counter
        counter += 1
        return x**2 + np.cos(y)

    res_foo = foo(x, y)
    res_bar = bar(x, y)

    assert np.allclose(res_foo, res_bar)
    assert counter == math.ceil(B / (batch_size or B)) + 1


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
        (
            np.arange(10 * 4 * 2).reshape(10, 4, 2),
            np.random.randn(10, 4, 2),
            None,
        ),
    ],
)
def test_batchify_on_method(x, y, batch_size):
    counter = 0
    B = x.shape[0]

    def foo(x: np.array, y: np.array):
        nonlocal counter
        counter += 1
        return x**2 + np.cos(y)

    class A:
        def __init__(self):
            self.batch_size = batch_size

        @batchify
        def bar(self, x: np.array, y: np.array):
            return foo(x, y)

    res_foo = foo(x, y)
    res_bar = A().bar(x, y)

    assert np.allclose(res_foo, res_bar)
    assert counter == math.ceil(B / (batch_size or B)) + 1


@pytest.mark.parametrize(
    "x,y,batch_size_",
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
        (
            np.arange(10 * 4 * 2).reshape(10, 4, 2),
            np.random.randn(10, 4, 2),
            None,
        ),
    ],
)
def test_batchify_on_class_method(x, y, batch_size_):
    counter = 0
    B = x.shape[0]

    def foo(x: np.array, y: np.array):
        nonlocal counter
        counter += 1
        return x**2 + np.cos(y)

    class A:
        batch_size = batch_size_

        @classmethod
        @batchify
        def bar(cls, x: np.array, y: np.array):
            return foo(x, y)

    res_foo = foo(x, y)
    res_bar = A.bar(x, y)

    assert np.allclose(res_foo, res_bar)
    assert counter == math.ceil(B / (batch_size_ or B)) + 1
