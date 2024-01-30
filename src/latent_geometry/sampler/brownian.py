import numpy as np

from latent_geometry.metric import Metric
from latent_geometry.sampler.abstract import Sampler


class BrownianSampler(Sampler):
    def __init__(self, metric: Metric) -> None:
        self.metric = metric

    def sample_gaussian(  # type: ignore
        self,
        means: np.ndarray,
        stds: np.ndarray,
        steps: int = 1,
        seed: int = 0,
        eigval_thold: float = 1e-3,
    ) -> np.ndarray:
        for _ in range(steps):
            means = self.sample_single(means, stds / steps, seed, eigval_thold)
        return means

    def sample_single(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        seed: int = 0,
        eigval_thold: float = 1e-3,
    ) -> np.ndarray:
        metric_matrices = self.metric.metric_matrix(means)
        inv_metrics = self.inv_trimmed(metric_matrices, eigval_thold)
        ind_samples = self._sample(means, stds, seed)
        return np.matmul(inv_metrics, ind_samples[..., None])[..., 0]

    @staticmethod
    def inv_trimmed(matrices: np.ndarray, eigval_thold) -> np.ndarray:
        U, S, Vh = np.linalg.svd(matrices, hermitian=True)
        S = np.where(S > eigval_thold, 1 / S, 0)

        B, D, _ = U.shape
        diag_S = np.zeros((B, D, D))
        rows, cols = np.diag_indices(D)
        diag_S[:, rows, cols] = S

        return Vh.transpose((0, 2, 1)) @ diag_S @ U.transpose((0, 2, 1))
