from abc import ABC, abstractmethod

import numpy as np


class Connection(ABC):
    """Class for affine connections.

    Notes
    -----
    https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/connection.py
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
        """Compute the geodesic ODE associated with the connection.

        Parameters
        ----------
        position : (D,) ndarray
            Position on the manifold.
        velocity : (D,) ndarray
            Velocity at the point on the manifold.

        Returns
        -------
        acceleration : (D,) ndarray
            Acceleration in the given state.
        """
        raise NotImplementedError  # TODO

    def exponential_map(
        self, tangent_vec: np.ndarray, base_point: np.ndarray
    ) -> np.ndarray:
        """Exponential map associated with the affine connection.

        Exponential map at base_point of tangent_vec computed by integration
        of the geodesic equation (initial value problem), using
        Christoffel symbols.

        Parameters
        ----------
        tangent_vec : (D,) ndarray
            Tangent vector at the base point.
        base_point : (D,) ndarray
            Point on the manifold.

        Returns
        -------
        end_point : (D,) ndarray
            Point on the manifold.
        """
        raise NotImplementedError  # TODO

    def logarithm_map(
        self,
        end_point: np.ndarray,
        base_point: np.ndarray,
    ) -> np.ndarray:
        """Logarithm map associated with the affine connection.

        Solve the boundary value problem associated with the geodesic equation
        using Christoffel symbols.

        Parameters
        ----------
        end_point : (D,) ndarray
            Point on the manifold.
        base_point : (D,) ndarray
            Point on the manifold.

        Returns
        -------
        tangent_vec : (D,) ndarray
            Tangent vector at the base point.
        """
        raise NotImplementedError  # TODO
