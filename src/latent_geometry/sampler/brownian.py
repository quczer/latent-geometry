import functools
from typing import Optional

import numpy as np

from latent_geometry.metric import Metric
from latent_geometry.sampler.abstract import Sampler
from latent_geometry.utils import project


class BrownianSampler(Sampler):
    MAX_SAMPLES = 100

    def __init__(self, metric: Metric) -> None:
        self.metric = metric

    def sample_gaussian(  # type: ignore
        self,
        mean: np.ndarray,
        std: float,
        steps: int = 1,
        seed: Optional[int] = None,
        eigval_thold: float = 1e-3,
    ) -> np.ndarray:
        return self.sample_gaussian_with_history(
            mean=mean, std=std, steps=steps, seed=seed, eigval_thold=eigval_thold
        )[-1]

    def sample_gaussian_with_history(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        steps: int = 1,
        seed: Optional[int] = None,
        eigval_thold: float = 1e-3,
        perp_alpha: float = 0.5,
    ) -> list[np.ndarray]:
        step_std = np.sqrt(std**2 / steps)
        means = [mean]
        cnt = 0
        while cnt < steps:
            if len(means) > self.MAX_SAMPLES:
                print("reached maximum number of samples")
                break
            vec_principal, vec_perp = self.sample_directions(
                means[-1], step_std, seed, eigval_thold
            )
            if np.linalg.norm(vec_principal) > 0:
                cnt += 1
                means.append(means[-1] + vec_principal)
            else:
                means.append(means[-1] + perp_alpha * vec_perp)

        return means

    def sample_directions(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        seed: Optional[int],
        eigval_thold: float = 1e-3,
    ) -> tuple[np.ndarray, np.ndarray]:
        metric_matrix = project(self.metric.metric_matrix)(mean)
        inv_metric_principal, inv_metric_perp = project(
            functools.partial(self.inv_split, eigval_thold=eigval_thold)
        )(metric_matrix)
        ind_sample = self._sample(np.zeros_like(mean), std, seed)
        vec_principal, vec_perp = [
            np.matmul(m, ind_sample[..., None])[..., 0]
            for m in (inv_metric_principal, inv_metric_perp)
        ]
        return vec_principal, vec_perp

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
        """Inverse then split a batch of Hermitian matrices into principal and perpendicular subspaces."""
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
