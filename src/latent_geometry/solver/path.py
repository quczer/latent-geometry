from typing import Callable

import numpy as np


class Path:
    """Time parametrized path.

    Methods
    -------
    __call__(t): (float,) -> (D,) ndarray
        Given t - time from [0, 1] interval, returns
        the corresponding point on the path.

    velocity(t): (float,) -> (D,) ndarray
        Given t - time from [0, 1] interval, returns
        the velocity at the corresponding point on the path.

    acceleration(t): (float,) -> (D,) ndarray
        Given t - time from [0, 1] interval, returns
        the acceleration at the corresponding point on the path.
    """

    def __init__(
        self,
        x_fun: Callable[[float], np.ndarray],
        v_fun: Callable[[float], np.ndarray],
        a_fun: Callable[[float], np.ndarray],
    ):
        self.x_fun = x_fun
        self.v_fun = v_fun
        self.a_fun = a_fun

    def __call__(self, t: float) -> np.ndarray:
        return self.x_fun(t)

    def velocity(self, t: float) -> np.ndarray:
        return self.v_fun(t)

    def acceleration(self, t: float) -> np.ndarray:
        return self.a_fun(t)
