import math

import numpy as np
import torch
import torch.nn as nn
from torch.func import jacfwd, jacrev

from latent_geometry.mapping.abstract import DerivativeMapping, MatrixMapping


class BaseTorchModelMapping(DerivativeMapping):
    def __init__(self, model: nn.Module, in_shape: tuple[int], out_shape: tuple[int]):
        self.model = model
        self.in_shape = in_shape
        self.out_shape = out_shape

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

    @property
    def in_dim(self) -> int:
        return math.prod(self.in_shape)

    @property
    def out_dim(self) -> int:
        return math.prod(self.out_shape)

    def _to_torch(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x).to(self._get_model_device()).float()

    @staticmethod
    def _to_numpy(x_tensor: torch.Tensor) -> np.ndarray:
        return x_tensor.detach().cpu().numpy()

    def _get_model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")


class TorchModelMapping(MatrixMapping, BaseTorchModelMapping):
    def metric_matrix_derivative(
        self, z: np.ndarray, ambient_metric_matrix: np.ndarray
    ) -> np.ndarray:
        z_torch = self._to_torch(z)
        A_torch = self._to_torch(ambient_metric_matrix)
        J_fn = jacrev(self._call_flat_model)

        def metrix_matrix(z_torch: torch.Tensor) -> torch.Tensor:
            J = J_fn(z_torch)
            return torch.mm(torch.mm(J.t(), A_torch), J)

        dM_torch = jacrev(metrix_matrix)(z_torch)
        return self._to_numpy(dM_torch)
