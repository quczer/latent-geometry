import numpy as np

from latent_geometry.sampler.abstract import Sampler


class NaiveSampler(Sampler):
    def sample_gaussian(
        self, means: np.ndarray, stds: np.ndarray, seed: int = 0
    ) -> np.ndarray:
        return self._sample(means, stds, seed)
