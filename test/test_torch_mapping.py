import numpy as np
import pytest
import torch

from latent_geometry.mapping import BaseTorchModelMapping, TorchModelMapping
from latent_geometry.metric import EuclideanMetric, ManifoldMetric
from latent_geometry.utils import project

ATOL = 1e-7


@pytest.fixture
def simple_net():
    """Returns torch.nn.Module mapping: (B, 1, 4, 4) -> (B, 128)"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 3)
            self.fc1 = nn.Linear(24, 32)
            self.fc2 = nn.Linear(32, 64)
            self.fc3 = nn.Linear(64, 128)

        def forward(self, x):
            x = F.sigmoid(self.conv1(x))
            x = torch.flatten(x, 1)
            x = F.sigmoid(self.fc1(x))
            x = F.sigmoid(self.fc2(x))
            x = self.fc3(x)
            return x

    return Net()


@pytest.mark.parametrize(
    "z",
    [np.linspace(1, 10, 16), np.random.randn(16)],
)
def test_equality_on_simple_net(z: np.ndarray, simple_net: torch.nn.Module):
    slow_metric = ManifoldMetric(
        BaseTorchModelMapping(
            simple_net,
            (-1, 1, 4, 4),
            (
                -1,
                128,
            ),
        ),
        EuclideanMetric(),
    )
    fast_metric = ManifoldMetric(
        TorchModelMapping(
            simple_net,
            (-1, 1, 4, 4),
            (
                -1,
                128,
            ),
        ),
        EuclideanMetric(),
    )
    slow_dM = project(slow_metric.metric_matrix)(z)
    fast_dM = project(fast_metric.metric_matrix)(z)

    np.testing.assert_allclose(slow_dM, fast_dM)


@pytest.mark.parametrize(
    "z",
    [np.linspace(1, 10, 16), np.random.randn(16)],
)
def test_equality_batching(z: np.ndarray, simple_net: torch.nn.Module):
    mapping = TorchModelMapping(
        simple_net,
        (-1, 1, 4, 4),
        (-1, 128),
    )
    z_batch = z[None, ...].repeat(2, 0)
    np.testing.assert_allclose(project(mapping)(z), mapping(z_batch)[0], atol=ATOL)
    np.testing.assert_allclose(
        project(mapping.jacobian)(z),
        mapping.jacobian(z_batch)[0],
        atol=ATOL,
    )
    np.testing.assert_allclose(
        project(mapping.second_derivative)(z),
        mapping.second_derivative(z_batch)[0],
        atol=ATOL,
    )
    np.testing.assert_allclose(
        project(mapping.euclidean_metric_matrix_derivative)(z),
        mapping.euclidean_metric_matrix_derivative(z_batch)[0],
        atol=ATOL,
    )


@pytest.mark.parametrize(
    "z",
    [np.linspace(1, 10, 16), np.random.randn(16)],
)
def test_equality_batching_on_permuted_net(z: np.ndarray, simple_net: torch.nn.Module):
    def __permuted_call(x):
        return simple_net(x.permute(2, 1, 3, 0)).permute(1, 0)

    mapping = TorchModelMapping(
        simple_net, (4, 1, -1, 4), (128, -1), call_fn=__permuted_call
    )
    z_batch = z[None, ...].repeat(2, 0)
    np.testing.assert_allclose(project(mapping)(z), mapping(z_batch)[0], atol=ATOL)
    np.testing.assert_allclose(
        project(mapping.jacobian)(z),
        mapping.jacobian(z_batch)[0],
        atol=ATOL,
    )
    np.testing.assert_allclose(
        project(mapping.second_derivative)(z),
        mapping.second_derivative(z_batch)[0],
        atol=ATOL,
    )
    np.testing.assert_allclose(
        project(mapping.euclidean_metric_matrix_derivative)(z),
        mapping.euclidean_metric_matrix_derivative(z_batch)[0],
        atol=ATOL,
    )
