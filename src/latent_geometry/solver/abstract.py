from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class SolverFailedException(Exception):
    pass


class ExponentialSolver(ABC):
    @abstractmethod
    def mark_path(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> Callable[[float], np.ndarray]:
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
    ) -> Callable[[float], np.ndarray]:
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
