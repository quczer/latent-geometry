from typing import Literal

import numpy as np

from latent_geometry.manifold.abstract import Manifold
from latent_geometry.mapping import Mapping
from latent_geometry.metric import ManifoldMetric, Metric
from latent_geometry.path import ManifoldPath
from latent_geometry.solver import BVPLogarithmSolver, IVPExponentialSolver
from latent_geometry.utils import project


class LatentManifold(Manifold):
    def __init__(
        self,
        mapping: Mapping,
        ambient_metric: Metric,
        solver_rtol: float = 1e-3,
        solver_atol: float = 1e-6,
        pullback_metric_eps: float = 1e-5,
        ivp_method: Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"] = "RK45",
        bvp_n_mesh_nodes: int = 2_000,
    ):
        self._metric = ManifoldMetric(
            mapping, ambient_metric, pullback_metric_eps=pullback_metric_eps
        )
        self._exp_solver = IVPExponentialSolver(
            method=ivp_method, rtol=solver_rtol, atol=solver_atol
        )
        # NOTE: careful, computation time can scale linearly with `n_mesh_nodes`
        self._log_solver = BVPLogarithmSolver(
            tolerance=solver_atol, n_mesh_nodes=bvp_n_mesh_nodes
        )
        self.flat_acc_fun = project(self.metric.acceleration)

    def shortest_path(self, z_a: np.ndarray, z_b: np.ndarray) -> ManifoldPath:
        self._check_if_vector(z_a=z_a, z_b=z_b)
        solver_path = self._log_solver.find_path(z_a, z_b, self.metric.acceleration)
        return ManifoldPath(
            solver_path.position,
            self.metric,
        )

    def geodesic(
        self, z: np.ndarray, velocity_vec: np.ndarray, length: float = 1.0
    ) -> ManifoldPath:
        self._check_if_vector(z=z, velocity_vec=velocity_vec)
        velocity = self._adjust_vector_magnitude(z, velocity_vec, length)
        solver_path = self._exp_solver.compute_path(z, velocity, self.flat_acc_fun)
        return ManifoldPath(
            solver_path.position,
            self.metric,
        )

    def set_solver_tols(self, tol: float) -> None:
        self._exp_solver.tolerance = tol
        self._log_solver.tolerance = tol

    @property
    def metric(self) -> ManifoldMetric:
        return self._metric

    def _adjust_vector_magnitude(
        self, base_point: np.ndarray, vec: np.ndarray, length: float
    ) -> np.ndarray:
        pullback_length = project(self.metric.vector_length)(
            tangent_vec=vec, base_point=base_point
        )
        vec_rescaled = vec / pullback_length * length
        return vec_rescaled

    def _check_if_vector(self, /, **kwargs: np.ndarray) -> None:
        for name, arr in kwargs.items():
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"{name} must be a numpy array")
            if len(arr.shape) != 1:
                raise ValueError(f"{name} must be a 1-dimensional vector")
