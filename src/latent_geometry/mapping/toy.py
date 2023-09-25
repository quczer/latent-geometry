import numpy as np
import torch
from torch import nn

from latent_geometry.mapping import TorchModelMapping


class _NHModel(nn.Module):
    def forward(self, in_: torch.Tensor) -> torch.Tensor:
        xs, ys = in_.split(1, dim=1)
        return torch.stack([xs, ys, torch.sqrt(1 - xs**2 - ys**2)])


class _SIModel(nn.Module):
    def forward(self, in_: torch.Tensor) -> torch.Tensor:
        theta, phi = in_.split(1, dim=1)
        return torch.stack(
            [
                torch.cos(phi) * torch.sin(theta),
                torch.sin(phi) * torch.sin(theta),
                torch.cos(theta),
            ],
        )

    @staticmethod
    def inv(vec: np.ndarray) -> np.ndarray:
        """Inverse of __call__; (x, y, z) -> (phi, theta)."""
        x, y, z = vec
        theta = np.arccos(z)
        phi = np.arccos(x / np.sin(theta))
        return np.array([theta, phi])


def create_northern_hemisphere_mapping() -> TorchModelMapping:
    return TorchModelMapping(model=_NHModel(), in_shape=(2,), out_shape=(3,))


def create_sphere_immersion() -> TorchModelMapping:
    return TorchModelMapping(model=_SIModel(), in_shape=(2,), out_shape=(3,))
