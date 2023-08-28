from abc import ABC, abstractmethod

import numpy as np

from latent_geometry.connection import Connection
from latent_geometry.mapping import EuclideanMatrixMapping, Mapping, MatrixMapping
from latent_geometry.metric.abstract import Metric
from latent_geometry.metric.euclidean import EuclideanMetric


class PullbackMetric(Connection, Metric, ABC):
    @property
    @abstractmethod
    def ambient_metric(self) -> Metric:
        """Ambient metric we pull back from."""

    @abstractmethod
    def metric_matrix_derivative(self, zs: np.ndarray) -> np.ndarray:
        r"""Compute mapping's second derivative tensor.

        Parameters
        ----------
        zs : (B, D) ndarray
            Batch of points from the domain - usually latent space.

        ambient_metric_matrices : (B, D', D') ndarray
            Batch of metric matrices from the co-domain.

        Returns
        -------
        dMs: (B, D, D, D) ndarray
            Derivative of the inner-product matrices of the domain, where the index
            k of the derivation is last: math:`mat_{bijk} = \partial_k g_{bij}`
        """

    def cometric_matrix(self, base_points: np.ndarray) -> np.ndarray:
        """Inner co-product matrix at the cotangent space at a base point.

        This represents the cometric matrix, i.e. the inverse of the
        metric matrix.

        Parameters
        ----------
        base_poinst : (B, D) ndarray
            Base point on the manifold.

        Returns
        -------
        (B, D, D) ndarray
            Inverse of the inner-product matrix.
        """
        metric_matrices = self.metric_matrix(base_points)
        cometric_matrices = np.linalg.inv(metric_matrices)
        return cometric_matrices

    def christoffels(self, base_points: np.ndarray) -> np.ndarray:
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


class MappingPullbackMetric(PullbackMetric, ABC):
    @property
    @abstractmethod
    def mapping(self) -> Mapping:
        """Map from latent to ambient space."""

    def metric_matrix(self, base_points: np.ndarray) -> np.ndarray:
        Js = self.mapping.jacobian(base_points)
        # can also be done on GPU if needed
        if isinstance(self.ambient_metric, EuclideanMetric):
            return np.einsum("bij,bil->bjl", Js, Js)
        ambient_points = self.mapping(base_points)
        As = self.ambient_metric.metric_matrix(ambient_points)
        return np.einsum("bij,bik,bkl->bjl", Js, As, Js)

    def metric_matrix_derivative(self, base_points: np.ndarray) -> np.ndarray:
        if isinstance(self.mapping, EuclideanMatrixMapping):
            if not isinstance(self.ambient_metric, EuclideanMetric):
                raise ValueError(
                    f"Can use EuclideanMatrixMapping only with EuclideanMetric, got {type(self.ambient_metric)}"
                )
            return self.mapping.euclidean_metric_matrix_derivative(base_points)
        elif isinstance(self.mapping, MatrixMapping):
            ambient_points = self.mapping(base_points)
            # for euclidean ambient spaces these are the same --> waste of memory
            ambient_matrices = self.ambient_metric.metric_matrix(ambient_points)

            return self.mapping.metric_matrix_derivative(base_points, ambient_matrices)
        else:
            ambient_points = self.mapping(base_points)
            Js = self.mapping.jacobian(base_points)  # B x D' x D
            Hs = self.mapping.second_derivative(base_points)  # B x D' x D x D
            As = self.ambient_metric.metric_matrix(ambient_points)  # B x D' x D'

            # let f: D -> D', then dMs has shape B x D x D x D and the compute time is O(B x D' x D**2)
            term_1 = np.einsum("brs,brik,bsj->bijk", As, Hs, Js)
            term_2 = np.einsum("brs,bsjk,bri->bijk", As, Hs, Js)
            return term_1 + term_2
