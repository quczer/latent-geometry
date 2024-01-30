from abc import ABC, abstractmethod

import numpy as np


class Sampler(ABC):
    @staticmethod
    def _sample(means: np.ndarray, stds: np.ndarray, seed: int = 0) -> np.ndarray:
        generator = np.random.default_rng(seed)
        return generator.normal(
            loc=means, scale=np.broadcast_to(stds[..., None], means.shape)
        )

    @abstractmethod
    def sample_gaussian(
        self, means: np.ndarray, stds: np.ndarray, seed: int
    ) -> np.ndarray:
        """Sample from an isotropic Gaussian distribution with given mean and standard deviation.

        Parameters
        ----------
        means : (B, D) array
            Batch of means.

        stds : (B,) array
            Batch of non-negative standard deviations.

        Returns
        -------
        samples : (B, D) array
            Batch of samples.
        """
