import numpy as np

from latent_geometry.mapping.abstract import (
    DerivativeMapping,
    EuclideanMatrixMapping,
    MatrixMapping,
)


class LinearMapping(EuclideanMatrixMapping, MatrixMapping, DerivativeMapping):
    """The linear Mapping."""

    def __init__(self, M: np.ndarray):
        """
        M : (D', D) np.ndarray
            a linear mapping R^D -> R^D'
        """
        self._M = M

    @property
    def in_dim(self) -> int:
        return self._M.shape[1]

    @property
    def out_dim(self) -> int:
        return self._M.shape[0]

    def _batched_M(self, B: int) -> np.ndarray:
        return np.repeat(self._M[None, ...], B, axis=0)

    def __call__(self, zs: np.ndarray) -> np.ndarray:
        BM = self._batched_M(zs.shape[0])
        return np.einsum("bij,bj->bi", BM, zs)

    def jacobian(self, zs: np.ndarray) -> np.ndarray:
        B, D = zs.shape
        return self._batched_M(B)

    def second_derivative(self, zs: np.ndarray) -> np.ndarray:
        B, D = zs.shape
        return np.zeros((B, 1, D, D))

    def metric_matrix_derivative(
        self, zs: np.ndarray, ambient_metric_matrices: np.ndarray
    ) -> np.ndarray:
        B, D = zs.shape
        return np.zeros((B, D, D, D))

    def euclidean_metric_matrix_derivative(self, zs: np.ndarray) -> np.ndarray:
        B, D = zs.shape
        return np.zeros((B, D, D, D))
