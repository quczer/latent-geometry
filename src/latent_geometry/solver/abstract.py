from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class ExponentialSolver(ABC):
    @abstractmethod
    def integrate_path(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> Callable[[float], np.ndarray]:
        """Compute the path given starting position and velocity following
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
            from [0, TODO] interval and returns the correspoding point of
            the path.
        """
        ...
