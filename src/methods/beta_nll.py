import torch
import torch.nn as nn
from typing import Literal
from src.methods import RegressionBase
from torch.nn.functional import gaussian_nll_loss

class BetaNLLHeteroscedasticRegression(RegressionBase):
    def __init__(self, model: nn.Module, beta: float = 0.):
        super().__init__(model=model)
        self.beta = beta


    def loss(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        target: torch.Tensor,
        reduction: Literal['mean', 'sum'] = 'sum'
    ):
        beta_scaling = (var**self.beta).detach()
        loss = beta_scaling*gaussian_nll_loss(
            mean, target, var=var, reduction='none', full=True
        )
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()