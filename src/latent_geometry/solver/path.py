from typing import Callable, Optional

import numpy as np


class Path:
    """Time parametrized path.

    Attributes
    ----------
    length: float
        The length of the path. It is the integral
        of velocity over [0, 1] time interval.

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

    _INTEGRATE_INTERVALS = 100

    def __init__(
        self,
        x_fun: Callable[[float], np.ndarray],
        v_fun: Callable[[float], np.ndarray],
        a_fun: Callable[[float], np.ndarray],
    ):
        self.x_fun = x_fun
        self.v_fun = v_fun
        self.a_fun = a_fun
        self._length: Optional[float] = None

    def __call__(self, t: float) -> np.ndarray:
        return self.x_fun(t)

    def velocity(self, t: float) -> np.ndarray:
        return self.v_fun(t)

    def acceleration(self, t: float) -> np.ndarray:
        return self.a_fun(t)

    @property
    def length(self) -> float:
        if self._length is None:
            self._length = self._integrate_length()
        return self._length

    def _integrate_length(self) -> float:
        len_ = 0.0
        dt = 1.0 / Path._INTEGRATE_INTERVALS
        for t in np.linspace(0.0, 1.0, Path._INTEGRATE_INTERVALS):
            len_ += float(np.linalg.norm(self.v_fun(t))) * dt
        return len_
