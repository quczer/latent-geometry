import numpy as np
import pytest

from latent_geometry.solver.logarithm import BVPLogarithmSolver


@pytest.fixture
def logarithm_solver():
    """Returns new instance of exponential solver with default method."""

    return BVPLogarithmSolver()


@pytest.mark.parametrize(
    "start_theta, final_theta",
    [(0.0, np.pi), (0.0, np.pi / 2), (1.0, 2.0), (3.0, 1.0)],
)
def test_on_unit_circle(start_theta, final_theta, logarithm_solver: BVPLogarithmSolver):
    """velocity is perpendicular to position, acceleration to velocity."""
    NUM_EVALS = 20

    x_start = np.array([np.cos(start_theta), np.sin(start_theta)])
    x_end = np.array([np.cos(final_theta), np.sin(final_theta)])

    def acceleration_fun(x, v):
        def circle_vector(x, v):
            v_proj_on_x = x * np.dot(x, v) / np.linalg.norm(v) / np.linalg.norm(x)
            v_proj_perp = v - v_proj_on_x
            return -x * np.linalg.norm(v_proj_perp) ** 2

        def x_displacement_correction(x, v):
            return x / np.linalg.norm(x) - x

        circle_term = circle_vector(x, v)
        x_penalty = x_displacement_correction(x, v)
        return circle_term - x_penalty * 10.0

    path = logarithm_solver.find_path(x_start, x_end, acceleration_fun)
    xs, _, _ = path.get_moments(NUM_EVALS)

    for x_t, theta_t in zip(
        xs,
        np.linspace(start_theta, final_theta, NUM_EVALS),
    ):
        expected_x = np.array([np.cos(theta_t), np.sin(theta_t)])
        assert np.allclose(x_t, expected_x, atol=0.01)
