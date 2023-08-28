from functools import partial, wraps
from typing import Callable, Optional, Protocol, TypeVar, overload

import numpy as np

# here I should probably use ParamSpec, but it can't
# be a proper bound and I don't know how to get around that
_T = TypeVar("_T", bound=Callable[..., np.ndarray])


class _HasBatchSize(Protocol):
    @property
    def batch_size(self) -> int:
        ...


def _batchify_decorator(fun: _T, batch_size: int) -> _T:
    @wraps(fun)
    def __wrapper(*arrays: np.ndarray) -> np.ndarray:
        B = arrays[0].shape[0]
        result_arrays = []
        for i in range(0, B, batch_size):
            res = fun(*(arr[i : i + batch_size, ...] for arr in arrays))
            result_arrays.append(res)
        return np.concatenate(result_arrays, axis=0, casting="no")

    return __wrapper  # type: ignore


def _batchify_decorator_method(fun: _T) -> _T:
    @wraps(fun)
    def __wrapper(inst: _HasBatchSize, /, *arrays: np.ndarray) -> np.ndarray:
        B = arrays[0].shape[0]
        result_arrays = []
        batch_size = getattr(inst, "batch_size") or B
        for i in range(0, B, batch_size):
            res = fun(inst, *(arr[i : i + batch_size, ...] for arr in arrays))
            result_arrays.append(res)
        return np.concatenate(result_arrays, axis=0, casting="no")

    return __wrapper  # type: ignore


@overload
def batchify(fun: _T) -> _T:
    ...


@overload
def batchify(batch_size: Optional[int]) -> Callable[[_T], _T]:
    ...


def batchify(__fun=None, /, *, batch_size: Optional[int] = None):  # type: ignore
    """
    Split inputs into `batch_size` batches, applies the function and stacks output.
    Assumes that all inputs have the first dimension `B` - the batch size.

    Examples
    --------
    >>> @batchify(batch_size=40)
    ... def foo(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ...     return x + 2 * y
    ...
    >>> class A:
    ...     def __init__(self, batch_size: int | None = None):
    ...         self.batch_size = batch_size # crucial to have this property
    ...
    ...     @batchify
    ...     def bar(self, x: np.ndarray) -> np.ndarray:
    ...         return np.sin(x)
    ...
    >>> class B:
    ...     batch_size = 20
    ...
    ...     @classmethod
    ...     @batchify
    ...     def bar(cls, x: np.array, y: np.array):
    ...         return np.abs(x - y)
    """
    # called with parens: @batchify(batch_size=...)
    if __fun is None:
        if batch_size is None:
            raise ValueError("must provide batch_size if called on standalone function")
        return partial(_batchify_decorator, batch_size=batch_size)

    # called w/o parens: @batchify
    # assume it decorates a method and try to infer the batch size
    return _batchify_decorator_method(__fun)


def project(__fun: _T, /) -> _T:
    """Project function that takes batches as inputs to one that accepts singles.

    Let `f: (x: (B, D), y: (B, D)) -> (B, D, D)`
    then `project(f): (x: (D,), y: (D,) -> (D, D)`

    Examples
    --------
    >>> def foo(xs):
    ...     return xs[:, 0]
    ...
    >>> bar = project(foo)
    >>> x = np.zeros((3, ))
    >>> foo(x)
    IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
    >>> bar(x)
    0.0
    """

    @wraps(__fun)
    def __projected_fun(*xs: np.ndarray, **ys: np.ndarray):
        return __fun(
            *(x[None, ...] for x in xs), **{k: y[None, ...] for k, y in ys.items()}
        )[0]

    return __projected_fun  # type: ignore
