import numpy as np

from latent_geometry.manifold.abstract import Manifold
from latent_geometry.mapping.abstract import Mapping
from latent_geometry.metric.abstract import Metric
from latent_geometry.metric.euclidean import EuclideanMetric
from latent_geometry.metric.manifold import ManifoldMetric
from latent_geometry.path import ManifoldPath
from latent_geometry.solver.exponential import IVPExponentialSolver
from latent_geometry.solver.logarithm import BVPLogarithmSolver


class LatentManifold(Manifold):
    def __init__(self, mapping: Mapping, ambient_metric: Metric):
        self.metric = ManifoldMetric(mapping, ambient_metric)
        self._euclidean_latent_metric = EuclideanMetric(mapping.in_dim)
        self._exp_solver = IVPExponentialSolver()
        self._log_solver = BVPLogarithmSolver()

    def geodesic(self, z_a: np.ndarray, z_b: np.ndarray) -> ManifoldPath:
        solver_path = self._log_solver.find_path(z_a, z_b, self.metric.acceleration)
        return ManifoldPath(
            solver_path.position,
            solver_path.velocity,
            self.metric,
            self._euclidean_latent_metric,
        )

    def path_given_direction(
        self, z: np.ndarray, velocity_vec: np.ndarray, length: float = 1.0
    ) -> ManifoldPath:
        velocity = self._adjust_vector_magnitude(z, velocity_vec, length)
        solver_path = self._exp_solver.compute_path(
            z, velocity, self.metric.acceleration
        )
        return ManifoldPath(
            solver_path.position,
            solver_path.velocity,
            self.metric,
            self._euclidean_latent_metric,
        )

    def _adjust_vector_magnitude(
        self, base_point: np.ndarray, vec: np.ndarray, length: float
    ) -> np.ndarray:
        pullback_length = self.metric.vector_length(
            tangent_vec=vec, base_point=base_point
        )
        return vec / pullback_length * length
