import numpy as np
import pytest

from latent_geometry.solver.exponential import IVPExponentialSolver


@pytest.fixture
def exponential_solver():
    """Returns new instance of exponential solver with default method."""

    return IVPExponentialSolver()


@pytest.mark.parametrize(
    "start_theta, final_theta",
    [(0.0, np.pi), (0.0, np.pi / 2), (1.0, 2.0), (3.0, 1.0)],
)
def test_on_unit_circle(
    start_theta, final_theta, exponential_solver: IVPExponentialSolver
):
    """velocity is perpendicular to position, acceleration to velocity."""
    NUM_EVALS = 20

    x = np.array([np.cos(start_theta), np.sin(start_theta)])
    v_dir = np.array([-np.sin(start_theta), np.cos(start_theta)])
    v = v_dir * (final_theta - start_theta)  # set the length of the path

    def acceleration_fun(x, v):
        return -x * np.linalg.norm(v) ** 2

    gamma = exponential_solver.mark_path(x, v, acceleration_fun)

    for t, theta_t in zip(
        np.linspace(0.0, 1.0, NUM_EVALS),
        np.linspace(start_theta, final_theta, NUM_EVALS),
    ):
        xt = gamma(t)
        expected_x = np.array([np.cos(theta_t), np.sin(theta_t)])
        assert np.allclose(xt, expected_x, atol=0.01)
