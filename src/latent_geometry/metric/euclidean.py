import numpy as np

from latent_geometry.metric.abstract import Metric
from latent_geometry.utils import batched_eye


class EuclideanMetric(Metric):
    def __init__(self, dimension: int):
        self.dimension = dimension

    def metric_matrix(self, base_points: np.ndarray) -> np.ndarray:
        B, D = base_points.shape
        return batched_eye(B, D)
