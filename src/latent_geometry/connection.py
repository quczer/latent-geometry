from abc import ABC, abstractmethod

import numpy as np


class Connection(ABC):
    """Class for affine connections.

    Notes
    -----
    Inspired by: https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/connection.py
    """

    @abstractmethod
    def christoffels(self, base_point: np.ndarray) -> np.ndarray:
        """Christoffel symbols associated with the connection.

        The contravariant index is on the first dimension.

        Parameters
        ----------
        base_point : (D,) ndarray
            Point on the manifold.

        Returns
        -------
        gamma : (D, D, D) ndarray
            Christoffel symbols, with the contravariant index on
            the first dimension.
        """

    def acceleration(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Compute the acceleration vector given position and velocity.

        Parameters
        ----------
        position : (D,) ndarray
            Position on the manifold.
        velocity : (D,) ndarray
            Velocity at the point on the manifold.

        Returns
        -------
        acceleration : (D,) ndarray
            Acceleration vector in the given state.
        """
        gamma = self.christoffels(position)
        acceleration = np.einsum("ijk,j,k->i", gamma, velocity, -velocity)
        return acceleration
