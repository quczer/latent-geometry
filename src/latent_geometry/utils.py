from functools import partial, wraps
from typing import Callable, Optional, Protocol, TypeVar, overload

import numpy as np

# here I should probably use ParamSpec, but it can't
# be a proper bound which is a bummer
_T = TypeVar("_T", bound=Callable[..., np.ndarray])


class _HasBatchSize(Protocol):
    @property
    def batch_size(self) -> int:
        ...


def _batchify_decorator(fun: _T, batch_size: int) -> _T:
    @wraps(fun)
    def wrapper(*arrays: np.ndarray) -> np.ndarray:
        B = arrays[0].shape[0]
        result_arrays = []
        for i in range(0, B, batch_size):
            res = fun(*(arr[i : i + batch_size, ...] for arr in arrays))
            result_arrays.append(res)
        return np.concatenate(result_arrays, axis=0, casting="no")

    return wrapper


def _batchify_decorator_method(fun: _T) -> _T:
    @wraps(fun)
    def wrapper(inst: _HasBatchSize, *arrays: np.ndarray) -> np.ndarray:
        B = arrays[0].shape[0]
        result_arrays = []
        batch_size = getattr(inst, "batch_size") or B
        for i in range(0, B, batch_size):
            res = fun(inst, *(arr[i : i + batch_size, ...] for arr in arrays))
            result_arrays.append(res)
        return np.concatenate(result_arrays, axis=0, casting="no")

    return wrapper


@overload
def batchify(fun: _T) -> _T:
    ...


@overload
def batchify(batch_size: Optional[int]) -> Callable[[_T], _T]:
    ...


def batchify(fun=None, /, *, batch_size: Optional[int] = None):
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
    if fun is None:
        if batch_size is None:
            raise ValueError("must provide batch_size if called on standalone function")
        return partial(_batchify_decorator, batch_size=batch_size)

    # called w/o parens: @batchify
    # assume it decorates a method and try to infer the batch size
    return _batchify_decorator_method(fun)
