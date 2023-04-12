from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class Manifold(ABC):
    @abstractmethod
    def compute_geodesic(
        self, z_a: np.ndarray, z_b: np.ndarray
    ) -> Callable[[float], np.ndarray]:
        """Compute the geodesic from z_a to z_b.

        Parameters
        ----------
        z_a : (D,) ndarray
            Start point on the manifold.
        z_b : (D,) ndarray
            End point on the manifold.

        Returns
        -------
        path : callable (float,) -> (D,) ndarray
            Time-parametrized path that is a function that takes a float
            from [0, 1] interval and returns the correspoding point of
            the geodesic on the manifold.

            `path(0.0) == z_a`

            `path(1.0) == z_b`
        """
        ...

    @abstractmethod
    def compute_path_given_direction(
        self, z: np.ndarray, velocity_vec: np.ndarray
    ) -> Callable[[float], np.ndarray]:
        """Compute the path on the manifold starting from z
        and following the direction velocity_vec.
        Magnitude of velocity_vec will affect path's parametrization.

        Parameters
        ----------
        z : (D,) ndarray
            Start point on the manifold.
        velocity_vec : (D,) ndarray
            Start velocity at point z.

        Returns
        -------
        path : callable (float,) -> (D,) ndarray
            Time-parametrized path that is a function that takes a float
            from [0, 1] interval and returns the correspoding point of
            the path on the manifold.

            `path(0.0) == z`
        """
        ...
