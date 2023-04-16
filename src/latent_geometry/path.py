from typing import Callable, Optional

import numpy as np


class Path:
    """Time parametrized path.

    Attributes
    ----------
    length : float
        The length of the path. It is the integral
        of velocity over [0, 1] time interval.

    Methods
    -------
    __call__(t) : (float,) -> (D,) ndarray
        Given t - time from [0, 1] interval, returns
        the corresponding point on the path.

    velocity(t) : (float,) -> (D,) ndarray
        Given t - time from [0, 1] interval, returns
        the velocity at the corresponding point on the path.

    acceleration(t) : (float,) -> (D,) ndarray
        Given t - time from [0, 1] interval, returns
        the acceleration at the corresponding point on the path,
        with corresponding velocity.
    """

    _INTEGRATE_INTERVALS = 100
    _N_PATH_POINTS = 30

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

    def get_moments(
        self, n_points: Optional[int] = None
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Compute position, velocity and acceleration on `n_points`
        evenly distributed (wrt. time) points of the path.

        Parameters
        ----------
        n_points : int, optional
        """

        if n_points is None:
            n_points = Path._N_PATH_POINTS

        xs, vs, accs = [], [], []
        for t in np.linspace(0.0, 1.0, n_points):
            xs.append(self(t))
            vs.append(self.velocity(t))
            accs.append(self.acceleration(t))
        return xs, vs, accs

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
