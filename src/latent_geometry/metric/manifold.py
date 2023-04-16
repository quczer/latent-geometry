from latent_geometry.mapping.abstract import Mapping
from latent_geometry.metric.abstract import MappingPullbackMetric, Metric


class ManifoldMetric(MappingPullbackMetric):
    def __init__(self, mapping: Mapping, ambient_metric: Metric):
        self._mapping = mapping
        self._ambient_metric = ambient_metric

    @property
    def mapping(self) -> Mapping:
        return self._mapping

    @property
    def ambient_metric(self) -> Metric:
        return self._ambient_metric
