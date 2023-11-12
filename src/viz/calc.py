from functools import partial

import numpy as np
from scipy.interpolate import splev, splprep

from latent_geometry.manifold import Manifold
from latent_geometry.path import Path


def create_radials(
    centre: np.ndarray,
    divisions: int,
    manifold: Manifold,
    length: float,
) -> list[Path]:
    lines = []
    for theta in np.linspace(0, 2 * np.pi, divisions + 1)[:-1]:
        dir_vector = np.array([np.cos(theta), np.sin(theta)])
        path = manifold.geodesic(centre, dir_vector, length)
        lines.append(path)
    return lines


def create_circles(radials: list[Path], n_circles: int) -> list[Path]:
    circles: list[Path] = []

    for timestamp in np.linspace(0.0, 1.0, n_circles + 1)[1:]:
        interpolate_points = np.vstack(
            [line(timestamp) for line in radials] + [radials[0](timestamp)]
        )
        circle = create_loop(interpolate_points)
        circles.append(circle)
    return circles


def create_loop(interpolants: np.ndarray) -> Path:
    tck, u = splprep(interpolants.T, s=0, per=1)
    loop = Path(partial(_eval_loop, tck=tck))
    return loop


def _eval_loop(t: float, tck: tuple) -> np.ndarray:
    x, y = splev(t, tck)
    return np.hstack([x, y])
