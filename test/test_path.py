import numpy as np
import pytest

from latent_geometry.metric import EuclideanMetric
from latent_geometry.path import ManifoldPath


@pytest.mark.parametrize(
    "theta",
    [0, np.pi / 2, np.pi],
)
def test_integration_on_the_circle(theta):
    def x_fun(t):
        return np.array([np.cos(theta * t), np.sin(theta * t)])

    def v_fun(t):
        return np.array([-np.sin(theta * t), np.cos(theta * t)]) * theta

    def a_fun(t):
        return np.array([-np.cos(theta * t), -np.sin(theta * t)]) * theta**2

    metric = EuclideanMetric()
    path = ManifoldPath(x_fun, metric)
    assert np.isclose(path.euclidean_length(), theta)
    assert np.isclose(path.manifold_length(), theta)
