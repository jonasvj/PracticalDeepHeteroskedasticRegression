import torch
from typing import Literal
from src.methods import RegressionBase
from torch.nn.functional import gaussian_nll_loss


class NaiveNLLHeteroscedasticRegression(RegressionBase):


    def loss(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        target: torch.Tensor,
        reduction: Literal['mean', 'sum'] = 'sum'
    ):
        return gaussian_nll_loss(
            mean, target, var=var, reduction=reduction, full=True
        )