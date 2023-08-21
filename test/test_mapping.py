import numpy as np
import pytest
import torch

from latent_geometry.mapping import BaseTorchModelMapping, TorchModelMapping
from latent_geometry.metric import EuclideanMetric, ManifoldMetric


@pytest.fixture(scope="module")
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
            (1, 1, 4, 4),
            (
                1,
                128,
            ),
        ),
        EuclideanMetric(128),
    )
    fast_metric = ManifoldMetric(
        TorchModelMapping(
            simple_net,
            (1, 1, 4, 4),
            (
                1,
                128,
            ),
        ),
        EuclideanMetric(128),
    )
    slow_dM = slow_metric.metric_matrix(z[None, :])
    fast_dM = fast_metric.metric_matrix(z[None, :])

    assert np.allclose(slow_dM, fast_dM)
