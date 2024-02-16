import functools

import numpy as np

from latent_geometry.metric import Metric
from latent_geometry.sampler.abstract import Sampler
from latent_geometry.utils import project


class BrownianSampler(Sampler):
    def __init__(self, metric: Metric) -> None:
        self.metric = metric

    def sample_gaussian(  # type: ignore
        self,
        mean: np.ndarray,
        std: float,
        steps: int = 1,
        seed: int = 0,
        eigval_thold: float = 1e-3,
    ) -> np.ndarray:
        step_std = np.sqrt(std**2 / steps)
        for _ in range(steps):
            mean = self.sample_single(mean, step_std, seed, eigval_thold)
        return mean

    def sample_gaussian_with_history(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        steps: int = 1,
        seed: int = 0,
        eigval_thold: float = 1e-3,
    ) -> list[np.ndarray]:
        step_std = np.sqrt(std**2 / steps)
        means = [mean]
        for _ in range(steps):
            means.append(self.sample_single(means[-1], step_std, seed, eigval_thold))
        return means

    def sample_single(  # ? TODO: fix std
        self,
        mean: np.ndarray,
        std: np.ndarray,
        seed: int = 0,
        eigval_thold: float = 1e-3,
    ) -> np.ndarray:
        metric_matrix = project(self.metric.metric_matrix)(mean)
        inv_metric = project(
            functools.partial(self.inv_trimmed, eigval_thold=eigval_thold)
        )(metric_matrix)
        ind_sample = self._sample(np.zeros_like(mean), std, seed)
        return mean + np.matmul(inv_metric, ind_sample[..., None])[..., 0]

    """
    robić 100 nieeksplorujących, kombinacja liniowa
    eskperyment:
    N peptydów - podobne
    100 trajektorii (zapamiętanych)
    tsne
    step: 0.1 - popróbować
    std: 1, 2, 5, 10
    dystanse levensteina - można wrzucic do tsne/umap
    histogram dystansów w ambiencie - ma być chi2
    """

    @staticmethod
    def inv_trimmed(matrices: np.ndarray, eigval_thold) -> np.ndarray:
        def __diagonalize(vecs: np.ndarray) -> np.ndarray:
            B, D, _ = U.shape
            diag_inv_S = np.zeros((B, D, D))
            rows, cols = np.diag_indices(D)
            diag_inv_S[:, rows, cols] = inv_S

        U, S, Vh = np.linalg.svd(matrices, hermitian=True)
        # prevent warnings: evaluation is eager
        inv_S = np.where(S > eigval_thold, 1 / np.maximum(S, eigval_thold), 0)

        B, D, _ = U.shape
        diag_inv_S = np.zeros((B, D, D))
        rows, cols = np.diag_indices(D)
        diag_inv_S[:, rows, cols] = inv_S

        return Vh.transpose((0, 2, 1)) @ diag_inv_S @ U.transpose((0, 2, 1))

        # U @ S @ Vh
        # Vh.T @ inv_S @ A.T
