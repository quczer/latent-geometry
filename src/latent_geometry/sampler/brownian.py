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

    def sample_single(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        seed: int = 0,
        eigval_thold: float = 1e-3,
    ) -> np.ndarray:
        metric_matrix = project(self.metric.metric_matrix)(mean)
        inv_metric_principal, inv_metric_perp = project(
            functools.partial(self.inv_split, eigval_thold=eigval_thold)
        )(metric_matrix)
        ind_sample = self._sample(np.zeros_like(mean), std, seed)
        return mean + np.matmul(inv_metric_principal, ind_sample[..., None])[..., 0]

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
    def inv_split(
        matrices: np.ndarray, eigval_thold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        _EPS = 1e-9

        def __diagonalize(vecs: np.ndarray) -> np.ndarray:
            B, D = vecs.shape
            diag = np.zeros((B, D, D))
            rows, cols = np.diag_indices(D)
            diag[:, rows, cols] = vecs
            return diag

        U, S, Vh = np.linalg.svd(matrices, hermitian=True)
        # prevent warnings: evaluation is eager
        inv_S_principal = np.where(S > eigval_thold, 1 / np.maximum(S, _EPS), 0)
        # avoid infinities
        inv_S_perp = np.where(S <= eigval_thold, 1 / np.maximum(S, _EPS), 0)
        inv_S_perp_norm = np.maximum(
            np.linalg.norm(inv_S_perp, axis=1, keepdims=True),
            np.full_like(inv_S_perp, _EPS),
        )
        inv_S_perp_normalized = inv_S_perp / inv_S_perp_norm

        diag_inv_S_principal = __diagonalize(inv_S_principal)
        diag_inv_S_perp = __diagonalize(inv_S_perp_normalized)

        return tuple(
            [
                Vh.transpose((0, 2, 1)) @ diag @ U.transpose((0, 2, 1))
                for diag in (diag_inv_S_principal, diag_inv_S_perp)
            ]
        )

        # U @ S @ Vh
        # Vh.T @ inv_S @ A.T
