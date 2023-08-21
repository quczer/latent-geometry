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
        r"""Compute Christoffel symbols of the Levi-Civita connection.

        The Koszul formula defining the Levi-Civita connection gives the
        expression of the Christoffel symbols with respect to the metric:
        :math:`\Gamma^k_{ij}(p) = \frac{1}{2} g^{lk}(
        \partial_i g_{jl} + \partial_j g_{li} - \partial_l g_{ij})`,
        where:

        - :math:`p` represents the base point, and
        - :math:`g` represents the Riemannian metric tensor.

        Note that the function computing the derivative of the metric matrix
        puts the index of the derivation last.

        Parameters
        ----------
        base_point : (B, D) ndarray
            Base point on the manifold.

        Returns
        -------
        gamma : (B, D, D, D) ndarray
            Christoffel symbols, where the contravariant index is second.
        """

    def acceleration(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Compute the acceleration vector given position and velocity.

        Parameters
        ----------
        position : (B, D) ndarray
            Position on the manifold.
        velocity : (B, D) ndarray
            Velocity at the point on the manifold.

        Returns
        -------
        acceleration : (B, D) ndarray
            Acceleration vector in the given state.
        """
        gamma = self.christoffels(position)
        acceleration = np.einsum("bijk,bj,bk->bi", gamma, velocity, -velocity)
        return acceleration
