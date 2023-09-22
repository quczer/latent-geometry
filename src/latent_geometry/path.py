import math
from typing import Callable, Final

import numpy as np

from latent_geometry.metric import EuclideanMetric, ManifoldMetric, Metric
from latent_geometry.utils import lift


class Path:
    """Time parametrized path.

    Methods
    -------
    __call__(t) : (float,) -> (D,) array
        Given t - time from [0, 1] interval, returns
        the corresponding point on the path.

    velocity(t) : (float,) -> (D,) array
        Given t - time from [0, 1] interval, returns
        the velocity at the corresponding point on the path.

    euclidean_length(t_start=0.0, t_end=0.0) : (float, float) -> float
        The length of the path calculated using the euclidean metric.
    """

    _INTEGRAL_DT = 0.0001

    def __init__(
        self,
        x_fun: Callable[[float], np.ndarray],
    ):
        self._x_fun: Final = x_fun
        self._euclidean_metric = EuclideanMetric()

    def __call__(self, t: float) -> np.ndarray:
        return self._x_fun(t)

    def velocity(self, t: float) -> np.ndarray:
        t_left, t_right = max(0, t - Path._INTEGRAL_DT), min(1, t + Path._INTEGRAL_DT)
        return (self(t_right) - self(t_left)) / (t_right - t_left)

    def euclidean_length(self, t_start: float = 0.0, t_end: float = 1.0) -> float:
        return self._integrate_length(self._euclidean_metric, t_start, t_end)

    def _integrate_length(self, metric: Metric, t_start: float, t_end: float) -> float:
        if t_end == t_start:
            return 0
        ts = np.arange(t_start, t_end + Path._INTEGRAL_DT, Path._INTEGRAL_DT)
        xs, vs = [], []
        for t1, t2 in zip(ts[:-1], ts[1:]):
            x1, x2 = self(t1), self(t2)
            v = (x2 - x1) / Path._INTEGRAL_DT
            x = (x1 + x2) / 2
            xs.append(x)
            vs.append(v)
        lengths = metric.vector_length(np.array(vs), np.array(xs)) * Path._INTEGRAL_DT
        return lengths.sum(axis=0)


class ManifoldPath(Path):
    """Time parametrized path.

    Methods
    -------
    __call__(t) : (float,) -> (D,) array
        Given t - time from [0, 1] interval, returns
        the corresponding point on the path.

    velocity(t) : (float,) -> (D,) array
        Given t - time from [0, 1] interval, returns
        the velocity at the corresponding point on the path.

    euclidean_length(t_start=0.0, t_end=0.0) : (float, float) -> float
        The length of the path calculated using the euclidean metric.

    manifold_length(t_start=0.0, t_end=0.0) : (float, float) -> float
        The length of the path calculated using the manifold metric.

    Attributes
    ----------
    ambient_path : Path
        Time parametrized path in the ambient space.
    """

    _AMBIENT_DT = 0.001

    def __init__(
        self,
        x_fun: Callable[[float], np.ndarray],
        manifold_metric: ManifoldMetric,
    ):
        super().__init__(x_fun)
        self._manifold_metric = manifold_metric
        self.ambient_path = self._create_ambient_path(x_fun, manifold_metric.mapping)

    def manifold_length(self, t_start: float = 0.0, t_end: float = 1.0) -> float:
        return self._integrate_length(self._manifold_metric, t_start, t_end)

    @staticmethod
    def _create_ambient_path(
        x_fun: Callable[[float], np.ndarray],
        mapping: Callable[[np.ndarray], np.ndarray],
    ) -> Path:
        # IMPROVE: interpolate better
        ts = np.arange(0, 1 + ManifoldPath._AMBIENT_DT, step=ManifoldPath._AMBIENT_DT)
        x_latent = lift(x_fun)(ts)
        x_ambient = mapping(x_latent)

        def x_ambient_fun(t: float) -> np.ndarray:
            idx = min(math.floor(t / ManifoldPath._AMBIENT_DT), len(x_ambient) - 2)
            lam = t - idx * ManifoldPath._AMBIENT_DT
            return x_ambient[idx] + lam * (x_ambient[idx + 1] - x_ambient[idx])

        return Path(x_ambient_fun)
