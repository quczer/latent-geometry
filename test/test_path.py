import numpy as np
import pytest

from latent_geometry.solver.path import Path


@pytest.mark.parametrize(
    "theta",
    [0, np.pi / 2, np.pi],
)
def test_integration_on_the_circle(theta):
    def x_fun(t):
        return np.array([np.cos(theta / (t + 1e-6)), np.sin(theta / (t + 1e-6))])

    def v_fun(t):
        return (
            np.array([-np.sin(theta / (t + 1e-6)), np.cos(theta / (t + 1e-6))]) * theta
        )

    def a_fun(t):
        return (
            np.array([-np.cos(theta / (t + 1e-6)), -np.sin(theta / (t + 1e-6))])
            * theta**2
        )

    path = Path(x_fun, v_fun, a_fun)
    assert np.isclose(path.length, theta)
