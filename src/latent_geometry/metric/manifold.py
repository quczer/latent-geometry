from latent_geometry.mapping import Mapping
from latent_geometry.metric.abstract import Metric
from latent_geometry.metric.constant import EuclideanMetric
from latent_geometry.metric.pullback import MappingPullbackMetric


class ManifoldMetric(MappingPullbackMetric):
    def __init__(
        self,
        mapping: Mapping,
        ambient_metric: Metric,
        pullback_metric_eps: float = 1e-5,
    ):
        super().__init__(matrix_eps=pullback_metric_eps)
        self._mapping = mapping
        self._ambient_metric = ambient_metric

    @property
    def mapping(self) -> Mapping:
        return self._mapping

    @property
    def ambient_metric(self) -> Metric:
        return self._ambient_metric


class EuclideanPullbackMetric(ManifoldMetric):
    def __init__(self, mapping: Mapping):
        super().__init__(mapping, EuclideanMetric())
