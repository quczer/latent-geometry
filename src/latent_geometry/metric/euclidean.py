import numpy as np

from latent_geometry.mapping.abstract import Mapping
from latent_geometry.metric.abstract import PullbackMetric


class EuclideanPullbackMetric(PullbackMetric):
    def __init__(self, mapping: Mapping):
        self.mapping = mapping

    def metric_matrix(self, base_point: np.ndarray) -> np.ndarray:
        J = self.mapping.jacobian(base_point)
        return J.T @ J  # TODO: dk if works

    def metric_matrix_derivative(self, base_point: np.ndarray) -> np.ndarray:
        return self.mapping.metric_matrix_derivative(base_point)  # TODO: dk if works
