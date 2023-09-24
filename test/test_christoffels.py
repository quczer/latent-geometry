import numpy as np
import pytest

from latent_geometry.mapping import create_sphere_immersion
from latent_geometry.metric import EuclideanPullbackMetric
from latent_geometry.utils import project


@pytest.mark.parametrize(
    "z",
    [
        np.array([np.pi / 3, 1.0]),
        np.array([np.pi / 4, 2.0]),
        np.array([np.pi / 6, 3.0]),
        np.array([-np.pi / 2, 4.0]),
    ],
)
def test_christoffels_on_the_sphere(z):
    def gt_tensor_fun(z_in):
        theta, phi = z_in
        christoffels = np.zeros((2, 2, 2))
        christoffels[1, 0, 1] = 1 / np.tan(theta)
        christoffels[1, 1, 0] = 1 / np.tan(theta)
        christoffels[0, 1, 1] = -0.5 * np.sin(2 * theta)
        return christoffels

    pullback_metric = EuclideanPullbackMetric(create_sphere_immersion())
    gt_tensor = gt_tensor_fun(z)
    computed_tensor = project(pullback_metric.christoffels)(z)
    assert np.allclose(gt_tensor, computed_tensor, atol=0.001)
