from typing import Optional

import numpy as np

from latent_geometry.sampler.abstract import Sampler


class NaiveSampler(Sampler):
    def sample_gaussian(
        self, mean: np.ndarray, std: float, seed: Optional[int] = None
    ) -> np.ndarray:
        return self._sample(mean, std, seed)