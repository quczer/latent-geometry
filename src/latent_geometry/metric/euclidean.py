import numpy as np

from latent_geometry.mapping.abstract import Mapping
from latent_geometry.metric.abstract import MappingPullbackMetric, Metric


class EuclideanMetric(Metric):
    def __init__(self, dimension: int):
        self.dimension = dimension

    def metric_matrix(self, base_point: np.ndarray) -> np.ndarray:
        return np.eye(N=self.dimension)


class EuclideanPullbackMetric(MappingPullbackMetric):
    def __init__(self, mapping: Mapping):
        self._mapping = mapping
        self._ambient_metric = EuclideanMetric(mapping.out_dim)

    @property
    def mapping(self) -> Mapping:
        return self._mapping

    @property
    def ambient_metric(self) -> Metric:
        return self._ambient_metric
