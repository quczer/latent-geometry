import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.func import jacfwd, jacrev, vmap

from latent_geometry.mapping.abstract import (
    DerivativeMapping,
    EuclideanMatrixMapping,
    MatrixMapping,
)
from latent_geometry.utils import batchify


class BaseTorchModelMapping(DerivativeMapping):
    def __init__(
        self,
        model: nn.Module,
        in_shape: tuple[int, ...],
        out_shape: tuple[int, ...],
        batch_size: Optional[int] = None,
    ):
        """Shapes should be without the batch dimension"""
        self.model = model
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.batch_size = batch_size

    @batchify
    def __call__(self, zs: np.ndarray) -> np.ndarray:
        B, _ = zs.shape
        zs_torch = self._to_torch(zs)
        xs_torch = self._call_flat_model(zs_torch, batch=True)
        xs = self._to_numpy(xs_torch)
        return xs.reshape(B, -1)

    @batchify
    def jacobian(self, zs: np.ndarray) -> np.ndarray:
        zs_torch = self._to_torch(zs)
        jacobian_torch = vmap(jacrev(self._call_flat_model))(zs_torch)
        return self._to_numpy(jacobian_torch)

    @batchify
    def second_derivative(self, zs: np.ndarray) -> np.ndarray:
        zs_torch = self._to_torch(zs)
        second_derivative_torch = vmap(jacfwd(jacrev(self._call_flat_model)))(zs_torch)
        return self._to_numpy(second_derivative_torch)

    def _call_flat_model(self, xs: torch.Tensor, batch: bool = False) -> torch.Tensor:
        """Reshapes data so that we can pretend that model's input and output is 2D (batch x latent_dim)."""
        in_shape = (-1, *self.in_shape)  # if batch else self.in_shape
        in_ = xs.reshape(in_shape)
        out_ = self.model(in_)

        return out_.reshape(-1, self.out_dim) if batch else out_.reshape(self.out_dim)

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


class TorchModelMapping(MatrixMapping, EuclideanMatrixMapping, BaseTorchModelMapping):
    @batchify
    def metric_matrix_derivative(
        self, zs: np.ndarray, ambient_metric_matrices: np.ndarray
    ) -> np.ndarray:
        zs_torch = self._to_torch(zs)
        As_torch = self._to_torch(ambient_metric_matrices)
        J_fn = jacrev(self._call_flat_model)

        def __metric_matrix(
            z_torch: torch.Tensor, A_torch: torch.Tensor
        ) -> torch.Tensor:
            J = J_fn(z_torch)
            return torch.mm(torch.mm(J.t(), A_torch), J)

        dMs_torch = vmap(jacrev(__metric_matrix))(zs_torch, As_torch)
        return self._to_numpy(dMs_torch)

    @batchify
    def euclidean_metric_matrix_derivative(self, zs: np.ndarray) -> np.ndarray:
        zs_torch = self._to_torch(zs)
        J_fn = jacrev(self._call_flat_model)

        def __metric_matrix(z_torch: torch.Tensor) -> torch.Tensor:
            J = J_fn(z_torch)
            return torch.mm(J.t(), J)

        dMs_torch = vmap(jacrev(__metric_matrix))(zs_torch)
        return self._to_numpy(dMs_torch)
