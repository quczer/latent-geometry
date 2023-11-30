import numpy as np

from latent_geometry.metric.abstract import Metric
from latent_geometry.utils import batched_eye


class ConstantMetric(Metric):
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    def metric_matrix(self, base_points: np.ndarray) -> np.ndarray:
        B, D = base_points.shape
        return self.matrix[None, ...].repeat(B, axis=0)


class EuclideanMetric(Metric):
    def metric_matrix(self, base_points: np.ndarray) -> np.ndarray:
        B, D = base_points.shape
        return batched_eye(B, D)


class SoftmaxEuclideanMetric(ConstantMetric):
    # def __init__(self, )
    def metric_matrix(self, base_points: np.ndarray) -> np.ndarray:
        B, D = base_points.shape
        return batched_eye(B, D)
