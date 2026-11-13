import torch
from typing import Literal
from src.methods import RegressionBase
from torch.nn.functional import gaussian_nll_loss


class FaithfulHeteroscedasticRegression(RegressionBase):


    def loss(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        target: torch.Tensor,
        reduction: Literal['mean', 'sum'] = 'sum'
    ):
        assert self.model.feature_extractor.detach_features
        loss = (
            0.5 * ((target - mean)**2) 
            + gaussian_nll_loss(mean.detach(), target, var=var, reduction='none', full=True)
        )
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()