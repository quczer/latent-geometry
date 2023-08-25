from functools import wraps
from typing import Callable, Optional, TypeVar

import numpy as np

_T = TypeVar("_T", bound=Callable[..., np.ndarray])


def batchify(*, batch_size: Optional[int]):
    """
    Split inputs into `batch_size` batches, applies the function and stacks output.
    Assumes all inputs have the first dimension `B` - the batch size.
    """

    def batchify_decorator(fun: _T) -> _T:
        @wraps(fun)
        def wrapper(*arrays: np.ndarray) -> np.ndarray:
            B = arrays[0].shape[0]
            result_arrays = []
            bs = batch_size or B
            for i in range(0, B, bs):
                res = fun(*(arr[i : i + bs, ...] for arr in arrays))
                result_arrays.append(res)
            return np.concatenate(result_arrays, axis=0, casting="no")

        return wrapper

    return batchify_decorator
