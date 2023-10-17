import math
from typing import Callable

import torch
import torch.optim.sgd
from torch.func import jacfwd


class InputSGDOptimizer:
    def __init__(
        self,
        param: torch.Tensor,
        mapping: Callable[[torch.Tensor], torch.Tensor],
        lr: float,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        assert isinstance(param, torch.Tensor)
        self.param = param
        self.mapping = mapping
        self.lr = lr

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            shape = self.param.shape
            g_inv_flat = self._calc_inv_matrix_flat(self.param, self.mapping)
            true_grad_flat = torch.mv(g_inv_flat, self.param.grad.reshape(-1))
            true_grad = true_grad_flat.reshape(shape)
            self.param.add_(true_grad, alpha=-self.lr)

        return loss

    def zero_grad(self):
        self.param.grad = None

    @staticmethod
    def _calc_inv_matrix_flat(
        x: torch.Tensor, mapping: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        shape = x.shape
        J = jacfwd(mapping)(x)
        # flatten input dims
        J_flat = J.reshape((*J.shape[: -len(shape)], math.prod(shape)))
        g_flat = torch.einsum("...i,...j->ij", J_flat, J_flat)
        return g_flat
