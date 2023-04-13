import numpy as np
import torch
import torch.nn as nn
from torch.func import jacfwd, jacrev

from latent_geometry.mapping.abstract import Mapping


class TorchModelMapping(Mapping):
    def __init__(self, model: nn.Module, in_shape: tuple[int]):
        self.model = model
        self.in_shape = in_shape

    def __call__(self, z: np.ndarray) -> np.ndarray:
        z_torch = self._to_torch(z)
        x_torch = self._call_flat_model(z_torch)
        x = self._to_numpy(x_torch)
        return x

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        z_torch = self._to_torch(z)
        jacobian_torch = jacrev(self._call_flat_model)(z_torch)
        return self._to_numpy(jacobian_torch)

    def second_derivative(self, z: np.ndarray) -> np.ndarray:
        z_torch = self._to_torch(z)
        second_derivative_torch = jacfwd(jacrev(self._call_flat_model))(z_torch)
        return self._to_numpy(second_derivative_torch)

    def _call_flat_model(self, x: torch.Tensor) -> torch.Tensor:
        """Reshapes data so that we can pretend that model's input and output is 1D."""
        in_ = x.reshape(self.in_shape)
        out_ = self.model(in_)
        return out_.reshape(-1)

    @staticmethod
    def _to_torch(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x).float()

    @staticmethod
    def _to_numpy(x_tensor: torch.Tensor) -> np.ndarray:
        return x_tensor.detach().numpy()
