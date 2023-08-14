from abc import ABC, abstractmethod

import numpy as np

from latent_geometry.connection import Connection
from latent_geometry.mapping.abstract import DerivativeMapping


class Metric(ABC):
    @abstractmethod
    def metric_matrix(self, base_point: np.ndarray) -> np.ndarray:
        """Metric matrix of the tangent space at a base point.

        Parameters
        ----------
        base_point : (D,) ndarray
            Base point.

        Returns
        -------
        (D, D) ndarray
            The inner-product matrix.
        """

    def inner_product(
        self,
        tangent_vec_a: np.ndarray,
        tangent_vec_b: np.ndarray,
        base_point: np.ndarray,
    ) -> float:
        """Inner product between two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : (D,) ndarray
            Tangent vector at a base point.
        tangent_vec_b : (D,) ndarray
            Tangent vector at a base point.
        base_point : (D,) ndarray
            Base point on the manifold.

        Returns
        -------
        float
            The inner-product.
        """
        inner_prod_matrix = self.metric_matrix(base_point)
        inner_prod = np.inner(tangent_vec_a, inner_prod_matrix @ tangent_vec_b)
        return inner_prod

    def vector_length(self, tangent_vec: np.ndarray, base_point: np.ndarray) -> float:
        """Length of a tangent vector at a base point.

        Parameters
        ----------
        tangent_vec : (D,) ndarray
            Tangent vector at a base point.
        base_point : (D,) ndarray
            Base point on the manifold.

        Returns
        -------
        float
            Length of the vector.
        """
        return np.sqrt(self.inner_product(tangent_vec, tangent_vec, base_point))


class PullbackMetric(Connection, Metric, ABC):
    @property
    @abstractmethod
    def ambient_metric(self) -> Metric:
        """Ambient metric we pull back from."""

    @abstractmethod
    def metric_matrix_derivative(self, base_point: np.ndarray) -> np.ndarray:
        r"""Compute derivative of the inner product matrix at a base point.

        Parameters
        ----------
        base_point : (D,) ndarray
            Base point on the manifold.

        Returns
        -------
        (D, D, D) ndarray
            Derivative of the inner-product matrix, where the index
            k of the derivation is last: math:`mat_{ijk} = \partial_k g_{ij}`.
        """

    def cometric_matrix(self, base_point: np.ndarray) -> np.ndarray:
        """Inner co-product matrix at the cotangent space at a base point.

        This represents the cometric matrix, i.e. the inverse of the
        metric matrix.

        Parameters
        ----------
        base_point : (D,) ndarray
            Base point on the manifold.

        Returns
        -------
        (D, D) ndarray
            Inverse of the inner-product matrix.
        """
        metric_matrix = self.metric_matrix(base_point)
        cometric_matrix = np.linalg.inv(metric_matrix)
        return cometric_matrix

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
        base_point : (D,) ndarray
            Base point on the manifold.

        Returns
        -------
        gamma : (D, D, D) ndarray
            Christoffel symbols, where the contravariant index is first.
        """
        cometric_mat_at_point = self.cometric_matrix(base_point)
        metric_derivative_at_point = self.metric_matrix_derivative(base_point)

        term_1 = np.einsum(
            "lk,jli->kij", cometric_mat_at_point, metric_derivative_at_point
        )
        term_2 = np.einsum(
            "lk,lij->kij", cometric_mat_at_point, metric_derivative_at_point
        )
        term_3 = -np.einsum(
            "lk,ijl->kij", cometric_mat_at_point, metric_derivative_at_point
        )

        christoffels = 0.5 * (term_1 + term_2 + term_3)
        return christoffels


class MappingPullbackMetric(PullbackMetric, ABC):
    @property
    @abstractmethod
    def mapping(self) -> DerivativeMapping:
        """Map from latent to ambient space."""

    def metric_matrix(self, base_point: np.ndarray) -> np.ndarray:
        ambient_point = self.mapping(base_point)
        J = self.mapping.jacobian(base_point)
        A = self.ambient_metric.metric_matrix(ambient_point)
        return J.T @ A @ J

    def metric_matrix_derivative(self, base_point: np.ndarray) -> np.ndarray:
        ambient_point = self.mapping(base_point)
        J = self.mapping.jacobian(base_point)
        H = self.mapping.second_derivative(base_point)
        A = self.ambient_metric.metric_matrix(ambient_point)

        term_1 = np.einsum("rs,rik,sj->ijk", A, H, J)  # TODO: very inefficient
        term_2 = np.einsum("rs,sjk,ri->ijk", A, H, J)
        return term_1 + term_2
