import numpy as np
import torch
from torch.func import jacrev

from latent_geometry.mapping.abstract import Mapping


class SphereImmersion(Mapping):
    """(phi, theta) -> (x, y, z) on the sphere in R^3."""

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.immerse(torch.tensor(z)).detach().numpy()

    def immerse(self, z: torch.Tensor) -> torch.Tensor:
        phi, theta = z
        return torch.stack(
            [
                torch.cos(phi) * torch.sin(theta),
                torch.sin(phi) * torch.sin(theta),
                torch.cos(theta),
            ],
        )

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        z_tensor = torch.tensor(z)
        return jacrev(self.immerse)(z_tensor).numpy()

    def second_derivative(self, z: np.ndarray) -> np.ndarray:
        z_tensor = torch.tensor(z)
        return jacrev(jacrev(self.immerse))(z_tensor).numpy()

    def metric_matrix_derivative(self, z: np.ndarray) -> np.ndarray:
        z_tensor = torch.tensor(z)

        def metric_matrix(x: torch.Tensor) -> torch.Tensor:
            J = jacrev(self.immerse)
            matrix = torch.mm(J(x).t(), J(x))
            return matrix

        matrix_derivative = jacrev(metric_matrix)(z_tensor)
        return matrix_derivative.numpy()

    @property
    def in_dim(self) -> int:
        return 2

    @property
    def out_dim(self) -> int:
        return 3
