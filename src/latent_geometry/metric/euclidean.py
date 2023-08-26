import numpy as np

from latent_geometry.metric.abstract import Metric


class EuclideanMetric(Metric):
    def __init__(self, dimension: int):
        self.dimension = dimension

    def metric_matrix(self, base_points: np.ndarray) -> np.ndarray:
        B, D = base_points.shape
        Is = np.zeros((B, D, D))
        diag_idx = np.arange(D)
        Is[:, diag_idx, diag_idx] = 1
        return Is
