from typing import Callable, Literal

import numpy as np
from scipy.integrate import solve_ivp

from latent_geometry.solver.abstract import ExponentialSolver, SolverFailedException


class IVPExponentialSolver(ExponentialSolver):
    def __init__(
        self,
        method: Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"] = "RK45",
    ):
        self.method = method

    def mark_path(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> Callable[[float], np.ndarray]:
        result = self._solve(position, velocity, acceleration_fun)
        if result.success:
            return self._wrap_path_solution(result.sol)
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
        y0 = self._pack_state(position, velocity)
        fun = self._create_fun(acceleration_fun)
        return solve_ivp(fun, t_span, y0, method=self.method, dense_output=True)

    @staticmethod
    def _create_fun(
        acceleration_fun: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        def fun(t: float, y: np.ndarray) -> np.ndarray:
            x, v = IVPExponentialSolver._unpack_state(y)
            a = acceleration_fun(x, v)
            y_prime = IVPExponentialSolver._pack_state(v, a)
            return y_prime

        return fun

    @staticmethod
    def _wrap_path_solution(
        ode_solution: Callable[[float], np.ndarray]
    ) -> Callable[[float], np.ndarray]:
        """In `ode_solution` values are both position and velocity - we only need position."""

        def gamma(t: float) -> np.ndarray:
            x, v = IVPExponentialSolver._unpack_state(ode_solution(t))
            return x

        return gamma

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
