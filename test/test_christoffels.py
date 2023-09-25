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


@pytest.mark.parametrize(
    "z",
    [
        np.array([np.pi / 6, 1.0]),
        # np.array([np.pi / 4, 2.0]),
        # np.array([np.pi / 6, 3.0]),
        # np.array([-np.pi / 2, 4.0]),
    ],
)
def test_christoffels_derivative_on_the_sphere(z):
    def gt_tensor_fun(z_in):
        theta, phi = z_in
        christoffels_der = np.zeros((2, 2, 2, 2))
        christoffels_der[1, 0, 1, 0] = -1 / np.sin(theta) ** 2
        christoffels_der[1, 1, 0, 0] = -1 / np.sin(theta) ** 2
        christoffels_der[0, 1, 1, 0] = -np.cos(theta)
        return christoffels_der

    pullback_metric = EuclideanPullbackMetric(create_sphere_immersion())
    gt_tensor = gt_tensor_fun(z)
    computed_tensor = project(pullback_metric.christoffels_derivative)(z)
    print(f"{project(pullback_metric.christoffels)(z).round(3)=}")
    print("gt humanize", gt_tensor.round(3).transpose(3, 0, 1, 2))
    print("computed humanize", computed_tensor.round(3).transpose(3, 0, 1, 2))
    assert np.allclose(gt_tensor, computed_tensor, atol=0.01)
