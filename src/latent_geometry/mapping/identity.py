import numpy as np

from latent_geometry.mapping.abstract import (
    DerivativeMapping,
    EuclideanMatrixMapping,
    MatrixMapping,
)
from latent_geometry.utils import batched_eye


class IdentityMapping(EuclideanMatrixMapping, MatrixMapping, DerivativeMapping):
    """The identity Mapping."""

    def __init__(self, dim: int):
        self._dim = dim

    @property
    def in_dim(self) -> int:
        return self._dim

    @property
    def out_dim(self) -> int:
        return self._dim

    def __call__(self, zs: np.ndarray) -> np.ndarray:
        return zs

    def jacobian(self, zs: np.ndarray) -> np.ndarray:
        B, D = zs.shape
        return batched_eye(B, D)

    def second_derivative(self, zs: np.ndarray) -> np.ndarray:
        B, D = zs.shape
        return np.zeros_like((B, D, D, D))

    def metric_matrix_derivative(
        self, zs: np.ndarray, ambient_metric_matrices: np.ndarray
    ) -> np.ndarray:
        B, D = zs.shape
        return np.zeros_like((B, D, D, D))

    def euclidean_metric_matrix_derivative(self, zs: np.ndarray) -> np.ndarray:
        B, D = zs.shape
        return np.zeros_like((B, D, D, D))
