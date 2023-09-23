import numpy as np
import torch
from torch.func import jacrev

from latent_geometry.mapping.abstract import DerivativeMapping


class _SphereImmersion(DerivativeMapping):
    """(phi, theta) -> (x, y, z) on the sphere in R^3."""

    def __call__(self, zs: np.ndarray) -> np.ndarray:
        return np.array([self.immerse(torch.tensor(z)).detach().numpy() for z in zs])

    @staticmethod
    def inv(vec: np.ndarray) -> np.ndarray:
        """Inverse of __call__; (x, y, z) -> (phi, theta)."""
        x, y, z = vec
        theta = np.arccos(z)
        phi = np.arccos(x / np.sin(theta))
        return np.array([phi, theta])

    def immerse(self, z: torch.Tensor) -> torch.Tensor:
        phi, theta = z
        return torch.stack(
            [
                torch.cos(phi) * torch.sin(theta),
                torch.sin(phi) * torch.sin(theta),
                torch.cos(theta),
            ],
        )

    def jacobian(self, zs: np.ndarray) -> np.ndarray:
        return np.array([jacrev(self.immerse)(torch.tensor(z)).numpy() for z in zs])

    def second_derivative(self, zs: np.ndarray) -> np.ndarray:
        return np.array(
            [jacrev(jacrev(self.immerse))(torch.tensor(z)).numpy() for z in zs]
        )

    def metric_matrix_derivative(self, zs: np.ndarray) -> np.ndarray:
        def __metric_matrix(x: torch.Tensor) -> torch.Tensor:
            J = jacrev(self.immerse)
            matrix = torch.mm(J(x).t(), J(x))
            return matrix

        return np.array([jacrev(__metric_matrix)(torch.tensor(z)).numpy() for z in zs])

    @property
    def in_dim(self) -> int:
        return 2

    @property
    def out_dim(self) -> int:
        return 3


def create_sphere_immersion() -> _SphereImmersion:
    return _SphereImmersion()
