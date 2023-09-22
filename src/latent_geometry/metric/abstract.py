from abc import ABC, abstractmethod

import numpy as np

from latent_geometry.utils import batched_eye

_EPS = 1e-4


class Metric(ABC):
    @abstractmethod
    def metric_matrix(self, base_points: np.ndarray) -> np.ndarray:
        """Metric matrix of the tangent space at a base point.

        Parameters
        ----------
        base_points : (B, D) ndarray
            Batch of base points.

        Returns
        -------
        (B, D, D) ndarray
            The inner-product matrices.
        """

    def cometric_matrix(self, base_points: np.ndarray) -> np.ndarray:
        """Inner co-product matrix at the cotangent space at a base point.

        This represents the cometric matrix, i.e. the inverse of the
        metric matrix.

        Parameters
        ----------
        base_points : (B, D) ndarray
            Base point on the manifold.

        Returns
        -------
        (B, D, D) ndarray
            Inverse of the inner-product matrix.
        """
        metric_matrices = self.metric_matrix(base_points)
        metric_matrices += _EPS * batched_eye(*base_points.shape)
        cometric_matrices = np.linalg.inv(metric_matrices)
        return cometric_matrices

    def inner_product(
        self,
        tangent_vec_a: np.ndarray,
        tangent_vec_b: np.ndarray,
        base_point: np.ndarray,
    ) -> float:
        """Inner product between two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : (B, D) ndarray
            Tangent vector at a base point.
        tangent_vec_b : (B, D) ndarray
            Tangent vector at a base point.
        base_point : (B, D) ndarray
            Base point on the manifold.

        Returns
        -------
        (B,) ndarry
            The inner-products.
        """
        inner_prod_matrices = self.metric_matrix(base_point)
        inner_prods = np.einsum(
            "bij,bi,bj->b", inner_prod_matrices, tangent_vec_a, tangent_vec_b
        )
        return inner_prods

    def vector_length(
        self, tangent_vec: np.ndarray, base_point: np.ndarray
    ) -> np.ndarray:
        """Length of a tangent vector at a base point.

        Parameters
        ----------
        tangent_vec : (B, D) ndarray
            Tangent vector at a base point.
        base_point : (B, D) ndarray
            Base point on the manifold.

        Returns
        -------
        (B,) ndarry
            Lengths of vectors.
        """
        return np.sqrt(self.inner_product(tangent_vec, tangent_vec, base_point))
