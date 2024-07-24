from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Sampler(ABC):
    def __init__(self, seed: Optional[int] = None) -> None:
        self.normal_gen = np.random.default_rng(seed)
        self.multivariate_gen = np.random.default_rng(seed)
        self.set_seed(seed)

    def set_seed(self, seed: Optional[int]) -> None:
        if seed is not None:  # !problematic
            self.normal_gen = np.random.default_rng(seed)
            self.multivariate_gen = np.random.default_rng(seed)

    def _sample_normal(
        self, mean: np.ndarray, std: float, seed: Optional[int] = None
    ) -> np.ndarray:
        self.set_seed(seed)
        sample = self.normal_gen.normal(loc=mean, scale=std)
        return sample

    def _sample_multivariate_normal(
        self, mean: np.ndarray, cov: np.ndarray, seed: Optional[int] = None
    ) -> np.ndarray:
        self.set_seed(seed)
        sample = self.multivariate_gen.multivariate_normal(mean, cov)
        return sample

    @abstractmethod
    def sample_gaussian(
        self, mean: np.ndarray, total_var: float, seed: Optional[int]
    ) -> np.ndarray:
        """Sample from an isotropic Gaussian distribution with a given mean and standard deviation.

        Parameters
        ----------
        mean : (D,) array

        total_var : float
            The total variance of the distribution.

        Returns
        -------
        sample : (D, ) array
        """
