import numpy as np

from latent_geometry.mapping.abstract import Mapping
from latent_geometry.metric.abstract import Metric
from latent_geometry.metric.manifold import ManifoldMetric


class EuclideanMetric(Metric):
    def __init__(self, dimension: int):
        self.dimension = dimension

    def metric_matrix(self, base_point: np.ndarray) -> np.ndarray:
        return np.eye(N=self.dimension)


class EuclideanPullbackMetric(ManifoldMetric):
    def __init__(self, mapping: Mapping):
        super().__init__(mapping, EuclideanMetric(mapping.out_dim))
