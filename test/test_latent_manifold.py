import numpy as np
import pytest
import torch
from torch import nn

from latent_geometry.manifold import LatentManifold
from latent_geometry.mapping import SphereImmersion, TorchModelMapping
from latent_geometry.metric import EuclideanMetric


@pytest.fixture
def sphere_manifold():
    """Returns a manifold from sphere immersion with euclidean ambient metric."""
    return LatentManifold(SphereImmersion(), EuclideanMetric(3))


@pytest.fixture
def hilly_2d_manifold():
    class Hilly2dNet(nn.Module):
        def forward(self, in_):
            x, y = np.split(in_.T, 2, axis=0)
            z = torch.max(torch.sin(x) + torch.cos(y), torch.tensor(0))
            return torch.stack([x, y, z])

    return LatentManifold(
        TorchModelMapping(Hilly2dNet(), (2,), (3,)), EuclideanMetric(3)
    )


def on_the_same_big_circle(xs: list[np.ndarray]) -> bool:
    def find_plane(x1, x2, x3) -> np.ndarray:
        A = np.vstack([x1, x2, x3])
        coefs = np.linalg.solve(A, np.zeros((3,)))
        return coefs

    plane_coefs = find_plane(*xs[:3])
    for x in xs:
        if not np.isclose(np.dot(plane_coefs, x), 0):
            return False
    return True


@pytest.mark.parametrize(
    "theta,phi,vec",
    [
        (0.0, 1.0, np.array([1.0, 0.0])),
        (-1.0, 1.0, np.array([1.0, -1.0])),
        (-1.0, 2.0, np.array([-3.0, -1.0])),
    ],
)
def test_exponential_mapping_on_the_sphere(
    theta, phi, vec, sphere_manifold: LatentManifold
):
    sphere_immersion = SphereImmersion()

    z = np.array([theta, phi])
    path = sphere_manifold.path_given_direction(z, vec)
    zs = [path(t) for t in np.linspace(0.0, 1.0)]
    xs = [sphere_immersion(z[None, :])[0] for z in zs]
    assert on_the_same_big_circle(xs)


@pytest.mark.parametrize(
    "amb_start,amb_end",
    [
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        (
            np.array([0.0, -1.0, 0.0]),
            np.array([1 / np.sqrt(3), -1 / np.sqrt(3), -1 / np.sqrt(3)]),
        ),
    ],
)
def test_logarithm_mapping_on_the_sphere(
    amb_start, amb_end, sphere_manifold: LatentManifold
):
    sphere_immersion = SphereImmersion()

    z_start, z_end = sphere_immersion.inv(amb_start), SphereImmersion.inv(amb_end)
    path = sphere_manifold.geodesic(z_start, z_end)
    zs = [path(t) for t in np.linspace(0.0, 1.0)]
    xs = [sphere_immersion(z[None, :])[0] for z in zs]
    assert on_the_same_big_circle(xs)


@pytest.mark.parametrize(
    "base_point,theta,vector_length",
    [
        (np.array([1, 0]), 1.0, 2),
        (np.array([-1, 0]), 1.0, 1),
        (np.array([1, -2]), -1.0, 3),
        (np.array([1, -2]), 0.0, 0.2),
    ],
)
def test_exponential_mapping_on_hilly_2d_graph(
    base_point, theta, vector_length, hilly_2d_manifold: LatentManifold
):
    direction = np.array([np.cos(theta), np.sin(theta)])
    path = hilly_2d_manifold.path_given_direction(base_point, direction, vector_length)
    assert np.allclose(path(0.0), base_point)
    assert np.isclose(
        path.manifold_length(), vector_length, rtol=0.01
    )  # TODO: precision is not there
    assert path.euclidean_length() <= path.manifold_length()
