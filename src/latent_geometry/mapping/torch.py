import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.func import jacfwd, vmap

from latent_geometry.mapping.abstract import (
    DerivativeMapping,
    EuclideanMatrixMapping,
    MatrixMapping,
)
from latent_geometry.utils import batchify


class BaseTorchModelMapping(DerivativeMapping):
    # IMPROVE: add an option for different plan of computing derivatives (jacrev/jacfwd)
    def __init__(
        self,
        model: nn.Module,
        in_shape: tuple[int, ...],
        out_shape: tuple[int, ...],
        batch_size: Optional[int] = None,
        call_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """Specify batch dimension with `-1`"""
        self.model = model
        self.call_fn = call_fn or model.__call__
        self.batch_size = batch_size
        self._init_shapes(in_shape, out_shape)
        assert not model.training, f"model must be in eval mode!"

    @batchify
    def __call__(self, zs: np.ndarray) -> np.ndarray:
        zs_torch = self._to_torch(zs)
        xs_torch = vmap(self._call_flat_model)(zs_torch)
        xs = self._to_numpy(xs_torch)
        return xs

    @batchify
    def call_shaped(self, zs: np.ndarray) -> np.ndarray:
        xs_flat = self(zs)
        xs_shaped = xs_flat.reshape(xs_flat.shape[0], *self.out_shape)  # -1 dim -> 1
        xs_shaped = xs_shaped.squeeze(self.out_batch_dim + 1)  # bath is now 1st
        return xs_shaped

    @batchify
    def jacobian(self, zs: np.ndarray) -> np.ndarray:
        zs_torch = self._to_torch(zs)
        jacobian_torch = vmap(jacfwd(self._call_flat_model))(zs_torch)
        return self._to_numpy(jacobian_torch)

    @batchify
    def second_derivative(self, zs: np.ndarray) -> np.ndarray:
        zs_torch = self._to_torch(zs)
        second_derivative_torch = vmap(jacfwd(jacfwd(self._call_flat_model)))(zs_torch)
        return self._to_numpy(second_derivative_torch)

    def _call_flat_model(self, x: torch.Tensor) -> torch.Tensor:
        """Reshapes data so that we can pretend that model's input and output is 1D (latent_dim,)."""
        in_ = x.reshape(self.in_shape)
        out_ = self.call_fn(in_)
        return out_.reshape(self.out_dim)

    @property
    def in_dim(self) -> int:
        return -math.prod(self.in_shape)

    @property
    def out_dim(self) -> int:
        return -math.prod(self.out_shape)

    def _to_torch(self, x: np.ndarray) -> torch.Tensor:
        if np.issubdtype(x.dtype, np.integer):
            torch_dtype = torch.int64
        else:
            torch_dtype = torch.float32
        return torch.tensor(x, dtype=torch_dtype).to(self._get_model_device())

    @staticmethod
    def _to_numpy(x_tensor: torch.Tensor) -> np.ndarray:
        return x_tensor.detach().cpu().numpy()

    def _get_model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _init_shapes(
        self,
        in_shape: tuple[int, ...],
        out_shape: tuple[int, ...],
    ) -> None:
        def __get_batch_dim(shape: tuple[int, ...], param_name: str) -> int:
            batch_dims = [i for i, d in enumerate(shape) if d == -1]
            if len(batch_dims) == 0:
                raise ValueError(
                    f"{param_name} = {shape} invalid - no batch dimension (-1)"
                )
            if len(batch_dims) > 1:
                raise ValueError(
                    f"{param_name} = {shape} invalid - multiple batch dimensions (-1)"
                )
            if any([i != -1 and i < 0 for i in shape]):
                raise ValueError(
                    f"{param_name} = {shape} invalid - negative dimension sizes"
                )
            return batch_dims[0]

        __get_batch_dim(in_shape, "in_shape")
        self.out_batch_dim = __get_batch_dim(out_shape, "out_shape")
        self.in_shape = in_shape
        self.out_shape = out_shape


class TorchModelMapping(MatrixMapping, EuclideanMatrixMapping, BaseTorchModelMapping):
    @batchify
    def metric_matrix_derivative(
        self, zs: np.ndarray, ambient_metric_matrices: np.ndarray
    ) -> np.ndarray:
        zs_torch = self._to_torch(zs)
        As_torch = self._to_torch(ambient_metric_matrices)
        J_fn = jacfwd(self._call_flat_model)

        def __metric_matrix(
            z_torch: torch.Tensor, A_torch: torch.Tensor
        ) -> torch.Tensor:
            J = J_fn(z_torch)
            return torch.mm(torch.mm(J.t(), A_torch), J)

        dMs_torch = vmap(jacfwd(__metric_matrix))(zs_torch, As_torch)
        return self._to_numpy(dMs_torch)

    @batchify
    def euclidean_metric_matrix_derivative(self, zs: np.ndarray) -> np.ndarray:
        zs_torch = self._to_torch(zs)
        J_fn = jacfwd(self._call_flat_model)

        def __metric_matrix(z_torch: torch.Tensor) -> torch.Tensor:
            J = J_fn(z_torch)
            return torch.mm(J.t(), J)

        dMs_torch = vmap(jacfwd(__metric_matrix))(zs_torch)
        return self._to_numpy(dMs_torch)
