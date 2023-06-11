from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class SolverResultPath:
    position: Callable[[float], np.ndarray]
    velocity: Callable[[float], np.ndarray]
    acceleration: Callable[[float], np.ndarray]

    _N_PATH_POINTS = 30

    def __call__(self, t: float) -> np.ndarray:
        return self.position(t)

    def get_moments(
        self, n_points: Optional[int] = None
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Compute position, velocity and acceleration on `n_points`
        evenly distributed (wrt. time) points of the path.

        Parameters
        ----------
        n_points : int, optional
        """

        if n_points is None:
            n_points = SolverResultPath._N_PATH_POINTS

        xs, vs, accs = [], [], []
        for t in np.linspace(0.0, 1.0, n_points):
            xs.append(self(t))
            vs.append(self.velocity(t))
            accs.append(self.acceleration(t))
        return xs, vs, accs
