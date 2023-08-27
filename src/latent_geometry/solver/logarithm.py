from typing import Callable

import numpy as np
from scipy.integrate import solve_bvp

from latent_geometry.solver.abstract import LogarithmSolver, SolverFailedException
from latent_geometry.solver.result import SolverResultPath


class BVPLogarithmSolver(LogarithmSolver):
    MAX_SCIPY_NODES = 10_000

    def __init__(self, n_mesh_nodes: int = 2, tolerance: float = 1e-3):
        assert n_mesh_nodes >= 2
        self.n_mesh_nodes = n_mesh_nodes
        self.tolerance = tolerance

    def find_path(
        self,
        start_position: np.ndarray,
        finish_position: np.ndarray,
        vectorized_acc_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> SolverResultPath:
        try:
            result = self._solve(start_position, finish_position, vectorized_acc_fun)
            if result.success:
                return self._create_result_path(result.sol, vectorized_acc_fun)
            else:
                raise SolverFailedException(result.message)
        except Exception as e:
            raise SolverFailedException from e

    def _solve(
        self,
        start_position: np.ndarray,
        finish_position: np.ndarray,
        acceleration_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        """Solve the boundary value problem.

        Need to rewrite our problem; scipy solve_bvp solves the following:

        ```
        dy / dt = f(t, y)
        bc(y(t0), y(tmax)) = 0
        ```
        let
        ```
        t0 := 0.0
        tmax := 1.0
        y(t) := (x(t), v(t))
        ```
        boundary condition
        ```
        y(t0) = y(0.0) = (start_position, ?some_v?)
        y(tmax) = y(1.0) = (finish_position, ?some_v?)
        ```
        then
        ```
        dy / dt = (v, a) = (v, a(x, v))
        ```
        so
        ```
        f(t, y) = f(t, (x, v)) = (v, acceleration_fun(x, v))
        bc(ya, yb) = (ya[0] - start_position, yb[0] - finish_position)

        ```

        Notes
        -----
        We will evaluate the solution in a couple of predefined points ti, therefore
        y is always of shape (2*D, k), where D is the dimension of our space,
        k - some number of mesh nodes the solver desires to evaluate in.

        Example:

        ```
        y = [
            [pos_00, pos_01, pos_02],
            [pos_10, pos_11, pos_12],
            [vel_00, vel_01, vel_02],
            [vel_10, vel_11, vel_12],
        ]
        ```
        for `D:=2, k:=3`
        """
        t_span = np.linspace(0.0, 1.0, self.n_mesh_nodes)
        points_on_initial_curve = self._create_initial_guess(
            start_position, finish_position
        )
        diff_eq = self._create_differential_equation(acceleration_fun)
        bc = self._create_boundary_condition(start_position, finish_position)
        return solve_bvp(
            fun=diff_eq,
            bc=bc,
            x=t_span,
            y=points_on_initial_curve,
            max_nodes=BVPLogarithmSolver.MAX_SCIPY_NODES,
            tol=self.tolerance,
        )

    def _create_initial_guess(
        self, start_position: np.ndarray, finish_position: np.ndarray
    ) -> np.ndarray:
        """Try to help the solver and propose points along linear path as the initial guess."""

        translation = finish_position - start_position
        ys = []
        for lam in np.linspace(0.0, 1.0, self.n_mesh_nodes):
            x = start_position + lam * translation
            v = translation
            yi = self._pack_state(x, v)
            ys.append(yi)
        return self._pack_mesh(*ys)

    @staticmethod
    def _create_differential_equation(
        acceleration_fun: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        def differential_eq(t: float, y: np.ndarray) -> np.ndarray:
            xs, vs = BVPLogarithmSolver._unpack_mesh2(y)
            accs = acceleration_fun(xs, vs)
            x_primes, v_primes = vs, accs
            return BVPLogarithmSolver._pack_mesh2(x_primes, v_primes)

        return differential_eq

    @staticmethod
    def _create_boundary_condition(
        start_position: np.ndarray, finish_position: np.ndarray
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
            xa, va = BVPLogarithmSolver._unpack_state(ya)
            xb, vb = BVPLogarithmSolver._unpack_state(yb)
            residual_a = xa - start_position
            residual_b = xb - finish_position
            return np.concatenate([residual_a, residual_b])

        return bc

    @staticmethod
    def _create_result_path(
        ode_solution: Callable[[float], np.ndarray],
        acceleration_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> SolverResultPath:
        def x_fun(t: float) -> np.ndarray:
            x, v = BVPLogarithmSolver._unpack_state(ode_solution(t))
            return x

        def v_fun(t: float) -> np.ndarray:
            x, v = BVPLogarithmSolver._unpack_state(ode_solution(t))
            return v

        def a_fun(t: float) -> np.ndarray:
            x, v = BVPLogarithmSolver._unpack_state(ode_solution(t))
            return acceleration_fun(x, v)

        return SolverResultPath(x_fun, v_fun, a_fun)

    @staticmethod
    def _pack_state(
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate((position, velocity))

    @staticmethod
    def _unpack_state(
        state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x, v = np.split(state, 2)
        return x, v

    @staticmethod
    def _pack_mesh(*states: np.ndarray) -> np.ndarray:
        return np.vstack(states).T

    @staticmethod
    def _unpack_mesh(mesh_state: np.ndarray) -> list[np.ndarray]:
        states = np.hsplit(mesh_state, mesh_state.shape[1])
        return [state.ravel() for state in states]

    @staticmethod
    def _unpack_mesh2(mesh_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(2*D, k) -> (k, D), (k, D) - position, velocity"""
        xs, vs = np.hsplit(mesh_state.T, 2)
        return xs, vs

    @staticmethod
    def _pack_mesh2(xs: np.ndarray, vs: np.ndarray) -> np.ndarray:
        """(k, D), (k, D) -> (2*D, k)"""
        return np.hstack((xs, vs)).T
