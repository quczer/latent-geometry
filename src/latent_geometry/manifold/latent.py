import numpy as np

from latent_geometry.manifold.abstract import Manifold
from latent_geometry.mapping.abstract import Mapping
from latent_geometry.metric.abstract import Metric
from latent_geometry.metric.manifold import ManifoldMetric
from latent_geometry.path import SolverResultPath
from latent_geometry.solver.exponential import IVPExponentialSolver
from latent_geometry.solver.logarithm import BVPLogarithmSolver


class LatentManifold(Manifold):
    def __init__(self, mapping: Mapping, ambient_metric: Metric):
        self.metric = ManifoldMetric(mapping, ambient_metric)
        self._exp_solver = IVPExponentialSolver()
        self._log_solver = BVPLogarithmSolver()

    def geodesic(self, z_a: np.ndarray, z_b: np.ndarray) -> SolverResultPath:
        path = self._log_solver.find_path(z_a, z_b, self.metric.acceleration)
        return path

    def path_given_direction(
        self, z: np.ndarray, velocity_vec: np.ndarray
    ) -> SolverResultPath:
        path = self._exp_solver.compute_path(z, velocity_vec, self.metric.acceleration)
        return path
