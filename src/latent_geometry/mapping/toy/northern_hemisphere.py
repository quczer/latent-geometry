import torch
from torch import nn

from latent_geometry.mapping import TorchModelMapping


class _NHModel(nn.Module):
    def forward(self, in_: torch.Tensor) -> torch.Tensor:
        xs, ys = in_.split(1, dim=1)
        return torch.stack([xs, ys, torch.sqrt(1 - xs**2 - ys**2)])


def create_northern_hemisphere_mapping() -> TorchModelMapping:
    return TorchModelMapping(model=_NHModel(), in_shape=(2,), out_shape=(3,))
