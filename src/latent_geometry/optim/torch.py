from typing import Literal

import torch
import torch.optim.sgd

from latent_geometry.optim.metric import TorchMetric


class InputSGDOptimizer:
    def __init__(
        self,
        param: torch.Tensor,
        metric: TorchMetric,
        lr: float,
        gradient_type: Literal["standard", "geometric", "retractive"] = "standard",
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if gradient_type not in ("standard", "geometric", "retractive"):
            raise ValueError(f"Invalid gradient_type: {gradient_type}")
        assert isinstance(param, torch.Tensor)

        self.param = param
        self.metric = metric
        self.lr = lr
        self.gradient_type = gradient_type

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            if self.gradient_type == "geometric":
                df = self.param.grad
                grad = self.metric.raise_index(df, self.param)
            elif self.gradient_type == "retractive":
                df = self.param.grad
                grad = self.metric.raise_index(df, self.param)
                euc_len = self.metric.euclidean_length(grad, self.param)
                man_len = self.metric.manifold_length(grad, self.param)
                grad.mul_(man_len / euc_len)
            else:
                grad = self.param.grad

            self.param.add_(grad, alpha=-self.lr)

        return loss

    def zero_grad(self):
        self.param.grad = None
