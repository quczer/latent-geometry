import numpy as np

from latent_geometry.mapping import Mapping
from latent_geometry.metric.abstract import Metric
from latent_geometry.metric.manifold import ManifoldMetric


class EuclideanMetric(Metric):
    def __init__(self, dimension: int):
        self.dimension = dimension

    def metric_matrix(self, base_points: np.ndarray) -> np.ndarray:
        B, D = base_points.shape
        Is = np.zeros((B, D, D))
        diag_idx = np.arange(D)
        Is[:, diag_idx, diag_idx] = 1
        return Is


class EuclideanPullbackMetric(ManifoldMetric):
    def __init__(self, mapping: Mapping):
        super().__init__(mapping, EuclideanMetric(mapping.out_dim))
