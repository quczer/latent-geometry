import math

import numpy as np
import pytest

from latent_geometry.utils import batchify, project


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

    def foo(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        nonlocal counter
        counter += 1
        return x**2 + np.cos(y)

    class A:
        def __init__(self):
            self.batch_size = batch_size

        @batchify
        def bar(self, x: np.ndarray, y: np.ndarray):
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

    def foo(x: np.ndarray, y: np.ndarray):
        nonlocal counter
        counter += 1
        return x**2 + np.cos(y)

    class A:
        batch_size = batch_size_

        @classmethod
        @batchify
        def bar(cls, x: np.ndarray, y: np.ndarray):
            return foo(x, y)

    res_foo = foo(x, y)
    res_bar = A.bar(x, y)

    assert np.allclose(res_foo, res_bar)
    assert counter == math.ceil(B / (batch_size_ or B)) + 1


@pytest.mark.parametrize(
    "x,y",
    [
        (np.zeros((3, 3)), np.ones((3, 3))),
        (np.zeros((10, 10, 3)), np.ones((10, 10, 3))),
    ],
)
def test_project(x, y):
    def foo(xs, ys):
        return xs[:, 0] + ys[:, 1]

    def bar(xs, ys, zs):
        return xs[:, 0] + ys[0] * zs[0]

    def qux():
        return np.ones(x.shape[0])

    assert np.allclose(project(foo)(x, y), x[0] + y[0])
    assert np.allclose(project(bar)(x, x, y), x[0] + x[0, 0] * y[0, 0])
    assert np.allclose(project(qux)(), 1)
