from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Sampler(ABC):
    @staticmethod
    def _sample(mean: np.ndarray, std: float, seed: Optional[int] = None) -> np.ndarray:
        generator = np.random.default_rng(seed)
        sample = generator.normal(loc=mean, scale=np.broadcast_to(std, mean.shape))
        return sample

    @abstractmethod
    def sample_gaussian(
        self, means: np.ndarray, stds: np.ndarray, seed: Optional[int]
    ) -> np.ndarray:
        """Sample from an isotropic Gaussian distribution with a given mean and standard deviation.

        Parameters
        ----------
        mean : (D,) array

        std : float

        Returns
        -------
        sample : (B, D) array
        """
