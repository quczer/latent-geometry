from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class SolverFailedException(Exception):
    pass


class Path:
    """Time parametrized path.

    Methods
    -------
    __call__(t): (float,) -> (D,) ndarray
        Given t - time from [0, 1] interval, returns
        the corresponding point on the path.

    velocity(t): (float,) -> (D,) ndarray
        Given t - time from [0, 1] interval, returns
        the velocity at the corresponding point on the path.

    acceleration(t): (float,) -> (D,) ndarray
        Given t - time from [0, 1] interval, returns
        the acceleration at the corresponding point on the path.
    """

    def __init__(
        self,
        x_fun: Callable[[float], np.ndarray],
        v_fun: Callable[[float], np.ndarray],
        a_fun: Callable[[float], np.ndarray],
    ):
        self.x_fun = x_fun
        self.v_fun = v_fun
        self.a_fun = a_fun

    def __call__(self, t: float) -> np.ndarray:
        return self.x_fun(t)

    def velocity(self, t: float) -> np.ndarray:
        return self.v_fun(t)

    def acceleration(self, t: float) -> np.ndarray:
        return self.a_fun(t)


class ExponentialSolver(ABC):
    @abstractmethod
    def mark_path(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> Path:
        """Compute the path given starting position and velocity, following
        the acceleration.

        Parameters
        ----------
        position : (D,) ndarray
            Position.
        velocity : (D,) ndarray
            Velocity at position.
        acceleration_fun : callable ((D,) ndarray, (D,) ndarray) -> (D,) ndarray
            Acceleration at any given position and velocity.

        Returns
        -------
        path : callable (float,) -> (D,) ndarray
            Time-parametrized path; function that takes a float
            from [0, 1] interval and returns the correspoding point of
            the path.

        Raises
        ------
        SolverFailedException
            If solution could not be found.
        """


class LogarithmSolver(ABC):
    @abstractmethod
    def find_path(
        self,
        start_position: np.ndarray,
        finish_position: np.ndarray,
        acceleration_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> Path:
        """Compute the path given starting position and finishing position,
        following the acceleration.

        Parameters
        ----------
        start_position : (D,) ndarray
        finish_position : (D,) ndarray
        acceleration_fun : callable ((D,) ndarray, (D,) ndarray) -> (D,) ndarray
            Acceleration at any given position and velocity.

        Returns
        -------
        path : callable (float,) -> (D,) ndarray
            Time-parametrized path; function that takes a float
            from [0, 1] interval and returns the correspoding point of
            the path.

        Raises
        ------
        SolverFailedException
            If solution could not be found.
        """
