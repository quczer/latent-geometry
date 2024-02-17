from typing import Optional

import numpy as np

from latent_geometry.metric import Metric
from latent_geometry.sampler.abstract import Sampler


class BrownianSampler(Sampler):
    def __init__(
        self, metric: Metric, max_samples: int = 1_000, seed: Optional[int] = None
    ) -> None:
        super().__init__(seed=seed)
        self.metric = metric
        self.max_samples = max_samples

    def sample_gaussian(  # type: ignore
        self,
        mean: np.ndarray,
        std: float,
        steps: int = 1,
        seed: Optional[int] = None,
        eigval_thold: float = 1e-3,
        perp_alpha: float = 0.5,
    ) -> np.ndarray:
        return self.sample_gaussian_with_history(
            mean=mean,
            std=std,
            steps=steps,
            seed=seed,
            eigval_thold=eigval_thold,
            perp_alpha=perp_alpha,
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
        # step_std = np.sqrt(std**2 / steps) # TODO: fix this
        self.set_seed(seed)
        step_std = std
        means = [mean]
        cnt = 0
        while cnt < steps:
            if len(means) > self.max_samples:
                print(f"reached maximum number of samples {self.max_samples}")
                break
            vec_principal, vec_perp = self.sample_directions(
                means[-1], step_std, eigval_thold
            )
            if np.linalg.norm(vec_principal) > 0:
                # print("found principal")
                cnt += 1
                means.append(means[-1] + vec_principal)
            else:
                # print("went perp")
                means.append(means[-1] + perp_alpha * vec_perp)

        return means

    def sample_directions(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        eigval_thold: float = 1e-3,
    ) -> tuple[np.ndarray, np.ndarray]:
        metric_matrix = self.metric.metric_matrix(mean[None, ...])[0]
        inv_metric_principal, inv_metric_perp = self.inv_split(
            metric_matrix, eigval_thold=eigval_thold
        )
        gauss_sample = self._sample(np.zeros_like(mean), std)
        vec_principal, vec_perp = [
            np.matmul(m, gauss_sample[..., None])[..., 0]
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
        matrix: np.ndarray, eigval_thold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inverse then split a batch of Hermitian matrices into principal and perpendicular subspaces."""
        _EPS = 1e-9

        def __diagonalize(vec: np.ndarray) -> np.ndarray:
            assert len(vec.shape) == 1
            return np.diag(vec)

        U, S, Vh = np.linalg.svd(matrix, hermitian=True)
        # prevent warnings: evaluation is eager
        inv_S_principal = np.where(S > eigval_thold, 1 / np.maximum(S, _EPS), 0)
        # print(f"principal components: {np.sum(S > eigval_thold)}")

        # avoid infinities
        inv_S_perp = np.where(S <= eigval_thold, 1 / np.maximum(S, _EPS), 0)
        inv_S_perp_norm = max(np.linalg.norm(inv_S_perp), _EPS)
        # same norm as an identity matrix
        inv_S_perp_normalized = inv_S_perp / inv_S_perp_norm * np.sqrt(S.shape[0])

        diag_inv_S_principal = __diagonalize(inv_S_principal)
        diag_inv_S_perp = __diagonalize(inv_S_perp_normalized)

        return tuple(
            [Vh.T @ diag @ U.T for diag in (diag_inv_S_principal, diag_inv_S_perp)]
        )

        # U @ S @ Vh
        # Vh.T @ inv_S @ A.T
