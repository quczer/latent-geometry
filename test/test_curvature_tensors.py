import numpy as np
import pytest

from latent_geometry.mapping import create_sphere_immersion
from latent_geometry.metric import EuclideanPullbackMetric
from latent_geometry.utils import project


@pytest.mark.parametrize(
    "z",
    [
        np.array([np.pi / 2, 0.0]),
        # np.array([1.0, np.pi / 2]),
        # np.array([2.0, np.pi]),
        # np.array([3.0, -np.pi]),
        # np.array([-1.0, 3.2]),
    ],
)
def test_riemann_tensor_on_the_sphere(z):
    def gt_tensor_fun(z):
        phi, theta = z
        riemann = np.zeros((2, 2, 2, 2))
        riemann[0, 1, 0, 1] = np.sin(theta) ** 2
        return riemann

    pullback_metric = EuclideanPullbackMetric(create_sphere_immersion())
    gt_tensor = gt_tensor_fun(z)
    computed_tensor = project(pullback_metric.riemann_tensor)(z)
    print(gt_tensor)
    print(computed_tensor)
    assert np.allclose(gt_tensor, computed_tensor)
