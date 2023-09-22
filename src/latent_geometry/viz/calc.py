from functools import partial
from typing import Callable

import numpy as np
from scipy.interpolate import splev, splprep

from latent_geometry.manifold import Manifold


def create_radials(
    centre: np.ndarray,
    divisions: int,
    manifold: Manifold,
    length: float,
) -> list[Callable[[float], np.ndarray]]:
    lines = []
    for theta in np.linspace(0, 2 * np.pi, divisions + 1)[:-1]:
        dir_vector = np.array([np.cos(theta), np.sin(theta)])
        path = manifold.path_given_direction(centre, dir_vector, length)
        lines.append(path)
    return lines


def create_circles(
    radials: list[Callable[[float], np.ndarray]], n_circles: int
) -> list[Callable[[float], np.ndarray]]:
    circles: list[Callable[[float], np.ndarray]] = []

    for timestamp in np.linspace(0.0, 1.0, n_circles + 1)[1:]:
        interpolate_points = np.vstack(
            [line(timestamp) for line in radials] + [radials[0](timestamp)]
        )
        tck, u = splprep(interpolate_points.T, s=0, per=1)
        circle = partial(_eval_circle, tck=tck)
        circles.append(circle)
    return circles


def _eval_circle(t: float, tck: tuple) -> np.ndarray:
    x, y = splev(t, tck)
    return np.hstack([x, y])
