from abc import ABC, abstractmethod

import numpy as np

from latent_geometry.metric import Connection
from latent_geometry.path import Path


class Manifold(ABC):
    @abstractmethod
    def geodesic(self, z_a: np.ndarray, z_b: np.ndarray) -> Path:
        """Compute the geodesic from `z_a` to `z_b`.

        Parameters
        ----------
        z_a : (D,) array
            Start point on the manifold.
        z_b : (D,) array
            End point on the manifold.

        Returns
        -------
        path : callable (float,) -> (D,) array
            Time-parametrized path; function that takes a float
            from [0, 1] interval and returns the correspoding point of
            the geodesic on the manifold.

            `path(0.0) == z_a`

            `path(1.0) == z_b`
        """

    @abstractmethod
    def path_given_direction(
        self, z: np.ndarray, velocity_vec: np.ndarray, length: float = 1.0
    ) -> Path:
        """Compute the path on the manifold starting from `z`
        and following the direction `velocity_vec`.

        Parameters
        ----------
        z : (D,) array
            Starting point on the manifold.
        velocity_vec : (D,) array
            Starting velocity direction at point z.
        length: float, default: 1.0

        Returns
        -------
        path : callable (float,) -> (D,) array
            Time-parametrized path that is a function that takes a float
            from [0, 1] interval and returns the correspoding point of
            the path on the manifold.

            `path(0.0) == z`
        """

    @property
    @abstractmethod
    def metric(self) -> Connection:
        """Metric on tangent space."""
