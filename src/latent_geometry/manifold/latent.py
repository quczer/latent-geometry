import numpy as np

from latent_geometry.manifold.abstract import Manifold
from latent_geometry.mapping import Mapping
from latent_geometry.metric import EuclideanMetric, ManifoldMetric, Metric
from latent_geometry.path import ManifoldPath
from latent_geometry.solver import BVPLogarithmSolver, IVPExponentialSolver


class LatentManifold(Manifold):
    def __init__(
        self,
        mapping: Mapping,
        ambient_metric: Metric,
        solver_tol: float = 1e-3,
        bvp_n_mesh_nodes: int = 2,
    ):
        self.metric = ManifoldMetric(mapping, ambient_metric)
        self._euclidean_latent_metric = EuclideanMetric(mapping.in_dim)
        self._exp_solver = IVPExponentialSolver(tolerance=solver_tol)
        # careful, computation time can scale linearly with `n_mesh_nodes`
        self._log_solver = BVPLogarithmSolver(
            tolerance=solver_tol, n_mesh_nodes=bvp_n_mesh_nodes
        )

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
