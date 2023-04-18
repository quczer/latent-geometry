import numpy as np
import pytest

from latent_geometry.manifold.latent import LatentManifold
from latent_geometry.mapping.sphere_immersion import SphereImmersion
from latent_geometry.metric.euclidean import EuclideanMetric


@pytest.fixture
def sphere_manifold():
    """Returns a manifold from sphere immersion with euclidean ambient metric."""
    return LatentManifold(SphereImmersion(), EuclideanMetric(3))


def random_point_on_the_sphere():
    p = np.random.randn(3)
    return p / np.linalg.norm(p)


def on_the_same_big_circle(xs: list[np.ndarray]) -> bool:
    def find_plane(x1, x2, x3) -> np.ndarray:
        A = np.vstack([x1, x2, x3])
        print(A)
        print(x1, x2, x3)
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
    zs, _, _ = path.get_moments()
    xs = [sphere_immersion(z) for z in zs]
    assert on_the_same_big_circle(xs)


@pytest.mark.parametrize(
    "amb_start,amb_end",
    [
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        (random_point_on_the_sphere(), random_point_on_the_sphere()),
        (random_point_on_the_sphere(), random_point_on_the_sphere()),
    ],
)
def test_logarithm_mapping_on_the_sphere(
    amb_start, amb_end, sphere_manifold: LatentManifold
):
    sphere_immersion = SphereImmersion()

    z_start, z_end = sphere_immersion.inv(amb_start), SphereImmersion.inv(amb_end)
    path = sphere_manifold.geodesic(z_start, z_end)
    zs, _, _ = path.get_moments()
    xs = [sphere_immersion(z) for z in zs]
    assert on_the_same_big_circle(xs)
