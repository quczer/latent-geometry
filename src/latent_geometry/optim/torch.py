import math
from typing import Callable, Literal

import torch
import torch.optim.sgd
from torch.func import jacfwd


class InputSGDOptimizer:
    def __init__(
        self,
        param: torch.Tensor,
        mapping: Callable[[torch.Tensor], torch.Tensor],
        lr: float,
        gradient_type: Literal["standard", "geometric"] = "standard",
        normalize_gradient: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if gradient_type not in ("standard", "geometric"):
            raise ValueError(f"Invalid gradient_type: {gradient_type}")
        assert isinstance(param, torch.Tensor)
        assert isinstance(normalize_gradient, bool)

        self.param = param
        self.mapping = mapping
        self.lr = lr
        self.gradient_type = gradient_type
        self.normalize_gradient = normalize_gradient

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            if self.gradient_type == "geometric":
                shape = self.param.shape
                g_flat = self._calc_metric_matrix_flat(self.param, self.mapping)
                g_inv_flat = torch.linalg.inv(g_flat)
                grad_flat = torch.mv(g_inv_flat, self.param.grad.reshape(-1))
                grad = grad_flat.reshape(shape)

                if self.normalize_gradient:
                    grad.mul_(
                        torch.linalg.norm(self.param.grad) / torch.linalg.norm(grad)
                    )
            else:
                grad = self.param.grad

            self.param.add_(grad, alpha=-self.lr)

        return loss

    def zero_grad(self):
        self.param.grad = None

    @staticmethod
    def _calc_metric_matrix_flat(
        x: torch.Tensor, mapping: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        shape = x.shape
        J = jacfwd(mapping)(x)
        # flatten input dims
        J_flat = J.reshape((*J.shape[: -len(shape)], math.prod(shape)))
        g_flat = torch.einsum("...i,...j->ij", J_flat, J_flat)
        return g_flat
