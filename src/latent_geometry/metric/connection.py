from abc import ABC, abstractmethod

import numpy as np

from latent_geometry.metric.abstract import Metric


class Connection(Metric, ABC):
    """Class for affine connections.

    Notes
    -----
    Inspired by: https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/connection.py
    """

    @abstractmethod
    def metric_matrix_derivative(self, zs: np.ndarray) -> np.ndarray:
        r"""Compute mapping's second derivative tensor.

        Parameters
        ----------
        zs : (B, D) array
            Batch of points from the domain - usually latent space.

        Returns
        -------
        dMs: (B, D, D, D) array
            Derivative of the inner-product matrices of the domain, where the index
            k of the derivation is last: math:`mat_{bijk} = \partial_k g_{bij}`
        """

    def christoffels(self, base_points: np.ndarray) -> np.ndarray:
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
        base_point : (B, D) array
            Base point on the manifold.

        Returns
        -------
        Gamma : (B, D, D, D) array
            Christoffel symbols, where the contravariant index is second.
        """
        cometric_mat_at_point = self.cometric_matrix(base_points)
        metric_derivative_at_point = self.metric_matrix_derivative(base_points)

        term_1 = np.einsum(
            "blk,bjli->bkij", cometric_mat_at_point, metric_derivative_at_point
        )
        term_2 = np.einsum(
            "blk,blij->bkij", cometric_mat_at_point, metric_derivative_at_point
        )
        term_3 = np.einsum(
            "blk,bijl->bkij", cometric_mat_at_point, metric_derivative_at_point
        )

        christoffels = 0.5 * (term_1 + term_2 - term_3)
        return christoffels

    def acceleration(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Compute the acceleration vector given position and velocity.

        Parameters
        ----------
        position : (B, D) array
            Position on the manifold.
        velocity : (B, D) array
            Velocity at the point on the manifold.

        Returns
        -------
        acceleration : (B, D) array
            Acceleration vector in the given state.
        """
        gamma = self.christoffels(position)
        acceleration = np.einsum("bijk,bj,bk->bi", gamma, velocity, -velocity)
        return acceleration


class ExtendedConnection(Connection, ABC):
    """Connection with more structure - e.g. Riemmann/Ricci curvature tensors."""

    @abstractmethod
    def christoffels_derivative(self, base_points: np.ndarray) -> np.ndarray:
        r"""
        Parameters
        ----------
        base_point : (B, D) array
            Base point on the manifold.

        Returns
        -------
        dGamma : (B, D, D, D, D) array
            Christoffel symbols derivative, where the contravariant index is second
            and the derivation index is last.

            `dGamma(bijkl) == \partial_l \Gamma^i_{jk}`
        """
