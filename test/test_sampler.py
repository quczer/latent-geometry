import numpy as np
import pytest

from latent_geometry.metric import EuclideanMetric
from latent_geometry.sampler import BrownianSampler, NaiveSampler


@pytest.mark.parametrize(
    "mu,total_var",
    [
        (np.array([0]), 1.0),
        (np.arange(10), 2.0),
        (np.random.randn(10), 3.0),
    ],
)
def test_naive(mu, total_var):
    sampler = NaiveSampler()
    sample = sampler.sample_gaussian(mu, total_var, seed=32)
    sample2 = np.random.default_rng(seed=32).normal(
        loc=mu, scale=np.sqrt(total_var / mu.size)
    )
    np.testing.assert_allclose(sample, sample2)


@pytest.mark.parametrize(
    "mu,total_var",
    [
        (np.array([0]), 1.0),
        (np.arange(10), 2.0),
        (np.random.randn(10), 3.0),
    ],
)
def test_simple_brownian(mu, total_var):
    naive = NaiveSampler()
    brownian = BrownianSampler(metric=EuclideanMetric())
    np.testing.assert_allclose(
        naive.sample_gaussian(mu, total_var, seed=52),
        brownian.sample_gaussian(mu, total_var, seed=52, steps=1, eigval_thold=0),
    )


@pytest.mark.parametrize(
    "A",
    [
        np.random.randn(25).reshape(5, 5),
    ],
)
def test_brownian_inverse(A):
    hermitian_A = A @ A.T
    inv_A, _, _ = BrownianSampler.inv_split(hermitian_A, eigval_thold=0)
    inv_A_true = np.linalg.inv(hermitian_A)
    np.testing.assert_allclose(inv_A, inv_A_true)
