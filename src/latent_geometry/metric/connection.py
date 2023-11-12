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
        :math:`\Gamma^i_{jk}(p) = \frac{1}{2} g^{il}(
        \partial_k g_{lj} + \partial_j g_{kl} - \partial_l g_{jk})`,
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
        cometric = self.cometric_matrix(base_points)
        metric_derivative = self.metric_matrix_derivative(base_points)
        term_1 = np.einsum("bil,bljk->bijk", cometric, metric_derivative)
        term_2 = np.einsum("bil,bklj->bijk", cometric, metric_derivative)
        term_3 = np.einsum("bil,bjkl->bijk", cometric, metric_derivative)

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


class RichConnection(Connection, ABC):
    """Connection with additional structure - e.g. Riemmann/Ricci curvature tensors."""

    @abstractmethod
    def christoffels_derivative(self, base_points: np.ndarray) -> np.ndarray:
        r"""
        Parameters
        ----------
        base_points : (B, D) array
            Base point on the manifold.

        Returns
        -------
        dGamma : (B, D, D, D, D) array
            Christoffel symbols derivative, where the contravariant index is second
            and the derivation index is last.

            `dGamma(bijkl) == \partial_l \Gamma^i_{jk}`
        """

    def riemann_tensor(self, base_points: np.ndarray) -> np.ndarray:
        r"""Compute the Riemann curvature tensor.

        The Riemann curvature tensor may be computed from the formula:
        :math:`R^i_{jkl}(p) = \partial_k \Gamma^i_{lj} - \partial_l \Gamma^i_{kj} +
        \Gamma^m_{lj} \Gamma^i_{km} - \Gamma^m_{kj} \Gamma^i_{lm}`, where

        - `\Gamma^i_{jk}` is the Christoffel symbol of the second kind.

        Parameters
        ----------
        base_points : (B, D) array
            Base point on the manifold.

        Returns
        -------
        R : (B, D, D, D, D) array
            Riemann (1, 3) curvature tensor, where the contravariant index is second.
        """
        gamma = self.christoffels(base_points)
        dGamma = self.christoffels_derivative(base_points)
        print("gamma", gamma.round(3))
        print("dGamma", dGamma.round(3))
        term_1 = np.einsum("biljk->bijkl", dGamma)
        term_2 = np.einsum("bikjl->bijkl", dGamma)
        term_3 = np.einsum("bmlj,bikm->bijkl", gamma, gamma)
        term_4 = np.einsum("bmkj,bilm->bijkl", gamma, gamma)
        return term_1 - term_2 + term_3 - term_4

    def ricci_tensor(self, base_points: np.ndarray) -> np.ndarray:
        r"""Compute the Ricci curvature tensor.

        The Ricci curvature tensor may be computed from the formula:
        :math:`R_{ij}(p) = R^k_{ikj}(p)`, where

        - `R^i_{jkl}` is the Riemann curvature tensor.

        Parameters
        ----------
        base_points : (B, D) array
            Base point on the manifold.

        Returns
        -------
        R : (B, D, D) array
            Ricci (0, 2) curvature tensor.
        """
        riemann = self.riemann_tensor(base_points)
        return np.einsum("bijkl->bjl", riemann)

    def ricci_scalar(self, base_points: np.ndarray) -> np.ndarray:
        r"""Compute the Ricci scalar.

        The Ricci scalar may be computed from the formula:
        :math:`R(p) = g^{ij} R_{ij}(p)`, where

        - `R_{ij}` is the Ricci tensor
        - `g^{ij}` is cometric tensor.

        Parameters
        ----------
        base_points : (B, D) array
            Base point on the manifold.

        Returns
        -------
        R : (B,) array
            The Ricci scalar.
        """
        ricci = self.ricci_tensor(base_points)
        cometric = self.cometric_matrix(base_points)
        return np.einsum("bij,bij->b", cometric, ricci)
