from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Sampler(ABC):
    def __init__(self, seed: Optional[int] = None) -> None:
        self.generator = np.random.default_rng(seed)
        self.set_seed(seed)

    def set_seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            self.generator = np.random.default_rng(seed)

    def _sample(
        self, mean: np.ndarray, std: float, seed: Optional[int] = None
    ) -> np.ndarray:
        self.set_seed(seed)
        sample = self.generator.normal(loc=mean, scale=std)
        return sample

    @abstractmethod
    def sample_gaussian(
        self, mean: np.ndarray, std: np.ndarray, seed: Optional[int]
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
