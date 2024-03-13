from typing import Optional

import numpy as np
from scipy.stats import chi

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
        total_var: float,
        steps: int = 1,
        seed: Optional[int] = None,
        eigval_thold: float = 1e-3,
        perp_eps: float = 1e-5,
    ) -> np.ndarray:
        history, outcomes = self.sample_gaussian_with_history(
            mean=mean,
            total_var=total_var,
            steps=steps,
            seed=seed,
            eigval_thold=eigval_thold,
            perp_eps=perp_eps,
        )
        return history[-1]

    def sample_gaussian_with_history(
        self,
        mean: np.ndarray,
        total_var: float,
        steps: int = 1,
        perp_eps: float = 1e-5,
        seed: Optional[int] = None,
        eigval_thold: float = 1e-3,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.set_seed(seed)
        step_var = total_var / steps
        means = [mean]
        cnt = 0
        n_components = []
        while cnt < steps:
            if len(means) > self.max_samples:
                print(f"reached maximum number of samples {self.max_samples}")
                break
            vec, n_comp = self.sample_directions(
                means[-1], step_var, eigval_thold, perp_eps=perp_eps
            )
            if n_comp > 0:
                cnt += 1
            means.append(means[-1] + vec)
            n_components.append(n_comp)
        return np.stack(means), np.array(n_components)

    def sample_directions(
        self,
        mean: np.ndarray,
        var: float,
        eigval_thold: float = 1e-3,
        perp_eps: float = 1e-5,
    ) -> tuple[np.ndarray, int]:
        metric_matrix = self.metric.metric_matrix(mean[None, ...])[0]
        cometric_star, n_components = self.hermitian_inv_star(
            metric_matrix, eigval_thold, perp_eps=perp_eps
        )
        success = n_components > 0
        if success:
            scale_squared = var
        else:
            scale_squared = var / chi.mean(df=n_components) ** 2
        sample = self._sample_multivariate_normal(
            np.zeros_like(mean), cometric_star * scale_squared
        )
        return sample, n_components

    @staticmethod
    def hermitian_inv_star(
        matrix: np.ndarray, eigval_thold: float, perp_eps: float
    ) -> tuple[np.ndarray, int]:
        _EPS = 1e-9
        NULL_RESULT = np.eye(matrix.shape[0]) * perp_eps, 0

        def __diagonalize(vec: np.ndarray) -> np.ndarray:
            assert len(vec.shape) == 1
            return np.diag(vec)

        U, S, Vh = np.linalg.svd(matrix, hermitian=True)
        # prevent warnings: evaluation is eager
        principal_mask = S > eigval_thold
        n_components = np.sum(principal_mask)
        inv_S_principal = np.where(principal_mask, 1 / np.maximum(S, _EPS), 0)

        if n_components == 0:
            print("no principal components")
            return NULL_RESULT

        # put epsilons on the diagonal
        inv_S_perp = np.where(~principal_mask, perp_eps, 0)

        diag = inv_S_principal + inv_S_perp
        diag = __diagonalize(diag)

        return Vh.T @ diag @ U.T, n_components
