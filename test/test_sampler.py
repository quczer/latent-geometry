import numpy as np
import pytest

from latent_geometry.metric import EuclideanMetric
from latent_geometry.sampler import BrownianSampler, NaiveSampler


@pytest.mark.parametrize(
    "mu,std",
    [
        (np.array([[0]]), np.array([1])),
        (np.arange(10)[None, :], np.array([2])),
        (np.random.randn(10, 4), np.arange(1, 11)),
    ],
)
def test_naive(mu, std):
    sampler = NaiveSampler()
    sample = sampler.sample_gaussian(mu, std, seed=32)
    sample2 = np.random.default_rng(seed=32).normal(loc=mu, scale=std[:, None])
    np.testing.assert_allclose(sample, sample2)


@pytest.mark.parametrize(
    "mu,std",
    [
        (np.array([[0]]), np.array([1])),
        (np.arange(10)[None, :], np.array([2])),
        (np.random.randn(10, 4), np.arange(1, 11)),
    ],
)
def test_simple_brownian(mu, std):
    naive = NaiveSampler()
    brownian = BrownianSampler(metric=EuclideanMetric())
    np.testing.assert_allclose(
        naive.sample_gaussian(mu, std, seed=52),
        brownian.sample_gaussian(mu, std, seed=52, steps=1, eigval_thold=0),
    )
