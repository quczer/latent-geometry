import numpy as np
import pytest
from torch.func import vmap

from latent_geometry.mapping import SphereImmersion, TorchModelMapping
from latent_geometry.metric import EuclideanPullbackMetric

ATOL = 1e-4


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


def random_16d_vector():
    """Returns random 16-vector (np.ndarray)"""
    return np.random.rand(16)


@pytest.mark.parametrize(
    "z",
    [
        np.array([np.pi / 2, 0.0]),
        np.array([1.0, 0.0]),
        np.array([2.0, 3.0]),
    ],
)
def test_metric_matrix_derivative_on_sphere_immersion(z):
    sphere_immersion = SphereImmersion()
    metric = EuclideanPullbackMetric(sphere_immersion)

    DM_gt = sphere_immersion.metric_matrix_derivative(z[None, :])
    DM_computed = metric.metric_matrix_derivative(z[None, :])
    assert np.allclose(DM_gt, DM_computed, atol=ATOL)


@pytest.mark.parametrize(
    "z",
    [
        np.arange(16),
        random_16d_vector(),
        random_16d_vector(),
    ],
)
def test_metric_matrix_on_torch_model(simple_net, z):
    metric = EuclideanPullbackMetric(
        TorchModelMapping(
            simple_net,
            (1, 4, 4),
            (128,),
        )
    )
    J = metric.mapping.jacobian(z[None, :])[0]
    M = metric.metric_matrix(z[None, :])[0]
    assert np.allclose(M, J.T @ J, atol=ATOL)


@pytest.mark.parametrize(
    "z",
    [
        np.arange(16),
        random_16d_vector(),
    ],
)
def test_metric_matrix_derivative_on_torch_model(simple_net, z):
    import torch
    from torch.func import jacfwd, jacrev

    torch_mapping = TorchModelMapping(simple_net, (1, 4, 4), (128,))
    metric = EuclideanPullbackMetric(torch_mapping)

    def compute_metric_matrix_torch(z_torch):
        J = jacrev(torch_mapping._call_flat_model)(z_torch)
        return torch.mm(J.t(), J)

    z_torch = torch_mapping._to_torch(z[None, :])
    DM_torch_gt = vmap(jacfwd(compute_metric_matrix_torch))(z_torch)

    DM_gt = torch_mapping._to_numpy(DM_torch_gt)[0]
    DM_computed = metric.metric_matrix_derivative(z[None, :])[0]

    assert np.allclose(DM_gt, DM_computed)
    assert np.abs(DM_computed).sum() > 0
