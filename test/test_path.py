import numpy as np
import pytest

from latent_geometry.mapping import IdentityMapping
from latent_geometry.metric import EuclideanMetric, ManifoldMetric
from latent_geometry.path import ManifoldPath


@pytest.mark.parametrize(
    "theta",
    [0, np.pi / 2, np.pi],
)
def test_integration_on_the_circle(theta):
    def x_fun(t):
        return np.array([np.cos(theta * t), np.sin(theta * t)])

    minifold_metric = ManifoldMetric(IdentityMapping(2), EuclideanMetric())
    manifold_path = ManifoldPath(x_fun, minifold_metric)
    ambient_path = manifold_path.ambient_path

    assert np.isclose(manifold_path.euclidean_length(), theta)
    assert np.isclose(manifold_path.manifold_length(), theta)
    for t in np.linspace(0, 1):
        assert np.allclose(ambient_path(t), manifold_path(t), atol=0.005)
