import numpy as np
import pytest

from latent_geometry.metric.euclidean import EuclideanMetric
from latent_geometry.path import ManifoldPath


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

    metric = EuclideanMetric(2)
    path = ManifoldPath(x_fun, v_fun, metric, metric)
    assert np.isclose(path.euclidean_length, theta)
    assert np.isclose(path.manifold_length, theta)
