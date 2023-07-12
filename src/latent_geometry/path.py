from typing import Callable, Final

import numpy as np

from latent_geometry.metric import Metric


class ManifoldPath:
    """Time parametrized path.

    Attributes
    ----------
    euclidean_length : float
        The length of the path wrt. euclidean latent metric.

    manifold_length : float
        The length of the path wrt. the metric of the manifold.

    Methods
    -------
    __call__(t) : (float,) -> (D,) ndarray
        Given t - time from [0, 1] interval, returns
        the corresponding point on the path.

    velocity(t) : (float,) -> (D,) ndarray
        Given t - time from [0, 1] interval, returns
        the velocity at the corresponding point on the path.
    """

    _INTEGRATE_INTERVALS = 100

    def __init__(
        self,
        x_fun: Callable[[float], np.ndarray],
        v_fun: Callable[[float], np.ndarray],
        manifold_metric: Metric,
        euclidean_metric: Metric,
    ):
        self._x_fun: Final = x_fun
        self._v_fun: Final = v_fun
        self._manifold_metric = manifold_metric
        self._euclidean_metric = euclidean_metric

    def __call__(self, t: float) -> np.ndarray:
        return self._x_fun(t)

    def velocity(self, t: float) -> np.ndarray:
        return self._v_fun(t)

    def euclidean_length(self, t_start: float = 0.0, t_end: float = 1.0) -> float:
        return self._integrate_length(self._euclidean_metric, t_start, t_end)

    def manifold_length(self, t_start: float = 0.0, t_end: float = 1.0) -> float:
        return self._integrate_length(self._manifold_metric, t_start, t_end)

    def _integrate_length(self, metric: Metric, t_start: float, t_end: float) -> float:
        len_ = 0.0
        # dt = (t_end - t_start) / ManifoldPath._INTEGRATE_INTERVALS
        ts = np.linspace(t_start, t_end, ManifoldPath._INTEGRATE_INTERVALS)
        for t1, t2 in zip(ts[:-1], ts[1:]):
            x1, x2 = self(t1), self(t2)
            v = x2 - x1
            len_ += metric.vector_length(v, x1)  # * dt
        return len_
