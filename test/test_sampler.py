import numpy as np
import pytest

from latent_geometry.metric import EuclideanMetric
from latent_geometry.sampler import BrownianSampler, NaiveSampler


@pytest.mark.parametrize(
    "mu,std",
    [
        (np.array([0]), 1.0),
        (np.arange(10), 2.0),
        (np.random.randn(10), 3.0),
    ],
)
def test_naive(mu, std):
    sampler = NaiveSampler()
    sample = sampler.sample_gaussian(mu, std, seed=32)
    sample2 = np.random.default_rng(seed=32).normal(loc=mu, scale=std)
    np.testing.assert_allclose(sample, sample2)


@pytest.mark.parametrize(
    "mu,std",
    [
        (np.array([0]), 1.0),
        (np.arange(10), 2.0),
        (np.random.randn(10), 3.0),
    ],
)
def test_simple_brownian(mu, std):
    naive = NaiveSampler()
    brownian = BrownianSampler(metric=EuclideanMetric())
    np.testing.assert_allclose(
        naive.sample_gaussian(mu, std, seed=52),
        brownian.sample_gaussian(mu, std, seed=52, steps=1, eigval_thold=0),
    )


@pytest.mark.parametrize(
    "As",
    [
        np.random.randn(50).reshape(2, 5, 5),
    ],
)
def test_brownian_inverse(As):
    hermitian_As = As @ As.transpose((0, 2, 1))
    inv_As, _ = BrownianSampler.inv_split(hermitian_As, eigval_thold=0)
    inv_As_true = np.linalg.inv(hermitian_As)
    np.testing.assert_allclose(inv_As, inv_As_true)
