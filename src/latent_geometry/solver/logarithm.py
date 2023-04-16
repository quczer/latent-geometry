from typing import Callable, Iterable

import numpy as np
from scipy.integrate import solve_bvp

from latent_geometry.path import Path
from latent_geometry.solver.abstract import LogarithmSolver, SolverFailedException


class BVPLogarithmSolver(LogarithmSolver):
    def __init__(self, n_mesh_nodes: int = 2):
        assert n_mesh_nodes >= 2
        self.n_mesh_nodes = n_mesh_nodes

    def find_path(
        self,
        start_position: np.ndarray,
        finish_position: np.ndarray,
        acceleration_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> Path:
        result = self._solve(start_position, finish_position, acceleration_fun)
        if result.success:
            return self._create_path(result.sol, acceleration_fun)
        else:
            raise SolverFailedException(result.message)

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
        """
        t_span = np.linspace(0.0, 1.0, self.n_mesh_nodes)
        y = self._create_y(start_position, finish_position)
        fun = self._create_fun(acceleration_fun)
        bc = self._create_boundary_condition(start_position, finish_position)
        return solve_bvp(fun, bc, x=t_span, y=y, max_nodes=10_000)

    def _create_y(
        self, start_position: np.ndarray, finish_position: np.ndarray
    ) -> np.ndarray:
        """Try to help the solver and propose linear path as the initial guess."""

        translation = finish_position - start_position
        ys = []
        for lam in np.linspace(0.0, 1.0, self.n_mesh_nodes):
            x = start_position + lam * translation
            v = translation
            yi = self._pack_state(x, v)
            ys.append(yi)
        return self._pack_mesh(*ys)

    @staticmethod
    def _create_fun(
        acceleration_fun: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        def fun(t: float, y: np.ndarray) -> np.ndarray:
            states = BVPLogarithmSolver._unpack_mesh(y)
            y_primes = []
            for yi in states:
                x, v = BVPLogarithmSolver._unpack_state(yi)
                a = acceleration_fun(x, v)
                y_prime = BVPLogarithmSolver._pack_state(v, a)
                y_primes.append(y_prime)
            return BVPLogarithmSolver._pack_mesh(*y_primes)

        return fun

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
    def _create_path(
        ode_solution: Callable[[float], np.ndarray],
        acceleration_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> Path:
        def x_fun(t: float) -> np.ndarray:
            x, v = BVPLogarithmSolver._unpack_state(ode_solution(t))
            return x

        def v_fun(t: float) -> np.ndarray:
            x, v = BVPLogarithmSolver._unpack_state(ode_solution(t))
            return v

        def a_fun(t: float) -> np.ndarray:
            x, v = BVPLogarithmSolver._unpack_state(ode_solution(t))
            return acceleration_fun(x, v)

        return Path(x_fun, v_fun, a_fun)

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
    def _unpack_mesh(mesh_state: np.ndarray) -> Iterable[np.ndarray]:
        states = np.hsplit(mesh_state, mesh_state.shape[1])
        return [state.ravel() for state in states]
