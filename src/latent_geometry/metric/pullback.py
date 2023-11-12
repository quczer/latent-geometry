from abc import ABC, abstractmethod

import numpy as np

from latent_geometry.mapping import (
    ChristoffelsDerivativeMapping,
    EuclideanMatrixMapping,
    Mapping,
    MatrixMapping,
)
from latent_geometry.metric.abstract import Metric
from latent_geometry.metric.connection import RichConnection
from latent_geometry.metric.euclidean import EuclideanMetric


class MappingPullbackMetric(RichConnection, ABC):
    @property
    @abstractmethod
    def ambient_metric(self) -> Metric:
        """Ambient metric we pull back from."""

    @property
    @abstractmethod
    def mapping(self) -> Mapping:
        """Map from latent to ambient space."""

    def metric_matrix(self, base_points: np.ndarray) -> np.ndarray:
        Js = self.mapping.jacobian(base_points)
        # NOTE: can also be done on GPU if needed
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

    def christoffels_derivative(self, base_points: np.ndarray) -> np.ndarray:
        if not isinstance(self.mapping, ChristoffelsDerivativeMapping):
            raise ValueError(
                f"Can use christoffels_derivative only with ChristoffelsDerivativeMapping, got {type(self.mapping)}"
            )
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise ValueError(
                f"Can use christoffels_derivative only with Euclidean ambient metric, got {type(self.ambient_metric)}"
            )
        return self.mapping.euclidean_christoffels_derivative(base_points)
