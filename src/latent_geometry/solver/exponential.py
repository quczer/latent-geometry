from typing import Callable, Literal

import numpy as np
from scipy.integrate import solve_ivp

from latent_geometry.solver.abstract import ExponentialSolver, SolverFailedException
from latent_geometry.solver.result import SolverResultPath


class IVPExponentialSolver(ExponentialSolver):
    def __init__(
        self,
        method: Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"] = "RK45",
        tolerance: float = 1e-3,
    ):
        self.method = method
        self.tolerance = tolerance

    def compute_path(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> SolverResultPath:
        result = self._solve(position, velocity, acceleration_fun)
        if result.success:
            return self._create_result_path(result.sol, acceleration_fun)
        else:
            raise SolverFailedException(result.message)

    def _solve(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        """Solve the initial value problem.

        Need to rewrite our problem; scipy solve_ivp solves the following:

        ```
        dy / dt = f(t, y)
        y(t0) = y0
        ```
        let
        ```
        t0 := 0
        y(t) := (x(t), v(t))
        ```
        then
        ```
        y(t0) = (x(t0), v(t0)) = (position, velocity)
        dy / dt = (v, a) = (v, a(x, v))
        ```
        so
        ```
        f(t, y) = f(t, (x, v)) = (v, acceleration_fun(x, v))
        ```
        """
        t_span = (0.0, 1.0)
        initial_state = self._pack_state(position, velocity)
        diff_eq = self._create_differential_equation(acceleration_fun)
        return solve_ivp(
            diff_eq,
            t_span,
            initial_state,
            method=self.method,
            dense_output=True,
            rtol=self.tolerance,
        )

    @staticmethod
    def _create_differential_equation(
        acceleration_fun: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        def differential_equation(t: float, state: np.ndarray) -> np.ndarray:
            x, v = IVPExponentialSolver._unpack_state(state)
            a = acceleration_fun(x, v)
            y_prime = IVPExponentialSolver._pack_state(v, a)
            return y_prime

        return differential_equation

    @staticmethod
    def _create_result_path(
        ode_solution: Callable[[float], np.ndarray],
        acceleration_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> SolverResultPath:
        def x_fun(t: float) -> np.ndarray:
            x, v = IVPExponentialSolver._unpack_state(ode_solution(t))
            return x

        def v_fun(t: float) -> np.ndarray:
            x, v = IVPExponentialSolver._unpack_state(ode_solution(t))
            return v

        def a_fun(t: float) -> np.ndarray:
            x, v = IVPExponentialSolver._unpack_state(ode_solution(t))
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
