import numpy as np

from latent_geometry.sampler.abstract import Sampler


class NaiveSampler(Sampler):
    def sample_gaussian(
        self, mean: np.ndarray, std: float, seed: int = 0
    ) -> np.ndarray:
        return self._sample(mean, std, seed)
