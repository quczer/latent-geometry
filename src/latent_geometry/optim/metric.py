import math
from typing import Callable

import torch
from torch.func import jacfwd


class TorchMetric:
    _EPS = 1e-5

    def __init__(self, mapping: Callable[[torch.Tensor], torch.Tensor]):
        self.mapping = mapping

    def inner_product(
        self, vec_a: torch.Tensor, vec_b: torch.Tensor, base_point: torch.Tensor
    ) -> torch.Tensor:
        vec_a_flat, vec_b_flat = (
            vec_a.reshape(-1),
            vec_b.reshape(-1),
        )
        g_flat = self._metric_matrix(base_point)
        return vec_a_flat.dot(g_flat.mv(vec_b_flat))

    def euclidean_length(
        self, vec: torch.Tensor, base_point: torch.Tensor
    ) -> torch.Tensor:
        vec_flat = vec.reshape(-1)
        return vec_flat.dot(vec_flat).sqrt()

    def manifold_length(
        self, vec: torch.Tensor, base_point: torch.Tensor
    ) -> torch.Tensor:
        return self.inner_product(vec, vec, base_point).sqrt()

    def lower_index(self, vec: torch.Tensor, base_point: torch.Tensor) -> torch.Tensor:
        shape = vec.shape
        g_flat = self._metric_matrix(base_point)
        return g_flat.mv(vec.reshape(-1)).reshape(shape)

    def raise_index(
        self, covec: torch.Tensor, base_point: torch.Tensor
    ) -> torch.Tensor:
        shape = covec.shape
        g_inv = self._cometric_matrix(base_point)
        return g_inv.mv(covec.reshape(-1)).reshape(shape)

    def _metric_matrix(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            shape = x.shape
            J = jacfwd(self.mapping)(x)
            # flatten input dims
            J_in_flat = J.reshape((*J.shape[: -len(shape)], math.prod(shape)))
            g_in_flat = torch.einsum("...i,...j->ij", J_in_flat, J_in_flat)
            return g_in_flat

    def _cometric_matrix(self, x: torch.Tensor) -> torch.Tensor:
        g = self._metric_matrix(x)
        g_invertible = g + self._EPS * torch.eye(g.shape[0], device=x.device)
        return torch.linalg.inv(g_invertible)
