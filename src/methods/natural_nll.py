import torch
from math import log, pi
from typing import Literal
from src.methods import RegressionBase
from src.utils import stable_softplus

C = -0.5*log(2*pi)

class NaturalNLLHeteroscedasticRegression(RegressionBase):

    def __init__(self, model, normalize_loss: bool = False):
        super().__init__(model)
        self.normalize_loss = normalize_loss


    def _preprocess_for_loss(self, mean, var, y, x):
        if self.normalize_loss and hasattr(self.model, 'normalize_targets'):
            mean = self.model.normalize_targets(mean, x.z, x.ptr)
            y    = self.model.normalize_targets(y,    x.z, x.ptr)
            var_scale = self.model.std ** 2
            if hasattr(self.model.variance_head, 'var_scale_factor'):
                var_scale = var_scale * stable_softplus(self.model.variance_head.var_scale_factor)
            var  = var / var_scale
        return mean, var, y


    def loss(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        target: torch.Tensor,
        reduction: Literal['mean', 'sum'] = 'sum'
    ):
        eta_1 = mean / var
        eta_2 = - 1 / (2*var)
        eta = torch.stack([eta_1, eta_2], dim=1)

        # Adapted from https://github.com/aleximmer/heteroscedastic-nn/blob/main/dependencies/asdl/asdfghjkl/loss.py
        target = torch.stack([target, target.square()], dim=1)
        target = target.to(eta.dtype)
        inner = torch.einsum('nk,nk->n', target, eta)
        log_A = eta[:, 0].square() / (4*eta[:, 1]) + 0.5*torch.log(-2*eta[:, 1])
        log_lik = len(mean)*C + inner.sum() + log_A.sum()
        nll = -log_lik

        if reduction == 'mean':
            return nll / len(mean)
        elif reduction == 'sum':
            return nll