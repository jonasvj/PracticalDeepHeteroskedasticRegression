import torch
import numpy as np
import torch.nn as nn
from tqdm import trange, tqdm
import uncertainty_toolbox as uct
from functools import partial
from torch.optim import Optimizer
import torch.distributions as dist
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from src.utils.early_stopping import EarlyStopping
from typing import Dict, Tuple, Literal, Union, Optional, Callable
from torch.nn.functional import gaussian_nll_loss
from src.utils.cal_metrics import (
    expected_calibration_error_regression, regression_ece_for_gaussian_mixture
)
from src.utils.utils import (
    stable_softplus, inverse_stable_softplus, transform_shapes_for_variance_ensemble
)


class RegressionBase(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.step = 0
        self.best_step = None
        self.max_num_steps = None


    def loss(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        target: torch.Tensor,
        reduction: Literal['mean', 'sum'] = 'sum'
    ):
        raise NotImplementedError


    def _preprocess_for_loss(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        y: torch.Tensor,
        x,
    ):
        """Hook for subclasses to transform mean/var/y before loss computation.
        By default returns inputs unchanged (physical space)."""
        return mean, var, y


    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        num_epochs: int = 1000,
        patience: int = 1000,
        min_epochs: int = 1000,
        es_criterion: str = 'mae',
        alt_loss: Optional[Callable] = None,
        disable_train: bool = False,
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        optimizer = optimizer(params=self.model.parameters())
        scheduler = scheduler(optimizer=optimizer)

        early_stopping = EarlyStopping(
            patience=patience,
            min_epochs=min_epochs,
        )

        stop = False
        epoch = 0
        pbar = tqdm(total=num_epochs)
        while epoch < num_epochs:
            loss_epoch = 0

            self.model.train() if not disable_train else self.model.eval()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                if hasattr(x, 'features') and hasattr(x, 'mean'):
                    mean, var = self.model(
                        x, use_var_scale_factor=True, mean=x.mean, features=x.features,
                    )
                else:
                    mean, var = self.model(x, use_var_scale_factor=True)

                mean_loss, var_loss, y_loss = self._preprocess_for_loss(mean, var, y, x)
                if alt_loss is None:
                    loss = self.loss(mean_loss, var_loss, y_loss, reduction='sum')
                else:
                    loss = alt_loss(mean_loss, var_loss, y_loss, reduction='sum')

                loss_step = loss / len(y)
                optimizer.zero_grad()
                loss_step.backward()
                optimizer.step()

                loss_epoch += loss.detach().item()

                self.step += 1

                if self.max_num_steps is not None and self.step >= self.max_num_steps:
                    stop = True
                    break

            # Update learning rate
            lr = scheduler.get_last_lr()[0]
            scheduler.step()

            loss_epoch /= len(train_loader.dataset)

            # Validation metrics
            means_val, vars_val, targets_val = self.predict(
                dataloader=val_loader, use_var_scale_factor=True
            )
            try:
                val_stats = self.compute_regression_stats(
                    means_val, vars_val, targets_val
                )
            except Exception as e:
                print('Error computing validation stats.')
                print(e)
                val_stats = {es_criterion: float('inf'), 'loss': float('inf')}

            pbar.set_postfix_str(
                f'loss={loss_epoch:.3e}, '
                f'val_loss={val_stats["loss"]:.3e}, '
                f'lr={lr:.3e}'
            )

            stop_es = early_stopping.check(self.model, val_stats[es_criterion], epoch, self.step)
            if stop or stop_es:
                print('Early stopping triggered.')
                print(
                    f'Best epoch: {early_stopping.best_epoch}; '
                    f'Best step: {early_stopping.best_step}; '
                    f'Best {es_criterion}: {early_stopping.best_loss}'
                )
                break

            epoch += 1
            pbar.update(1)

        pbar.close()

        best_state_dict = (
            early_stopping.best_model if early_stopping.best_model is not None
            else self.model.state_dict()
        )
        self.best_step = (
            early_stopping.best_step if early_stopping.best_step is not None
            else self.step
        )
        self.model.load_state_dict(best_state_dict)


    def calibrate_with_scale_factor(
        self,
        cal_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        num_epochs: int = 10000,
    ):
        means_cal, vars_cal, targets_cal = self.predict(
            cal_loader, use_var_scale_factor=False
        )

        scale_factor = nn.Parameter(
            inverse_stable_softplus(torch.tensor(1.0, device=means_cal.device)),
            requires_grad=True,
        )

        optimizer = optimizer(params=[scale_factor])
        scheduler = scheduler(optimizer=optimizer)

        if isinstance(optimizer, torch.optim.LBFGS):
            def nll_loss_func():
                optimizer.zero_grad()
                scaled_vars = stable_softplus(scale_factor)*vars_cal
                nll = gaussian_nll_loss(
                    means_cal, targets_cal, var=scaled_vars, full=True, reduction='mean'
                )
                nll.backward()
                return nll

        pbar = trange(num_epochs)
        for epoch in pbar:

            if isinstance(optimizer, torch.optim.LBFGS):
                nll = optimizer.step(nll_loss_func)
            else:
                pos_scale_factor = stable_softplus(scale_factor)
                scaled_vars = pos_scale_factor*vars_cal

                nll = gaussian_nll_loss(
                    means_cal, targets_cal, var=scaled_vars, full=True, reduction='mean'
                )

                optimizer.zero_grad()
                nll.backward()
                optimizer.step()

            # Update learning rate
            lr = scheduler.get_last_lr()[0]
            scheduler.step()

            self.step += 1

            pbar.set_postfix_str(
                f'cal_loss={nll.detach().item():.3e}, '
                f'var_scale_factor={scale_factor.item():.3e}, '
                f'lr={lr:.3e}'
            )

        # Insert optimized value into model
        copied = False
        for name, param in self.model.named_parameters():
            if 'var_scale_factor' in name:
                with torch.no_grad():
                    param.copy_(torch.tensor(scale_factor.item(), device=param.device))
                copied = True
                break
        if not copied:
            raise ValueError('Scale factor not copied correctly.')


    def calibrate_with_variance_head(
        self,
        cal_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        num_epochs: int = 1000,
        patience: int = 1000,
        min_epochs: int = 1000,
        fit_var_scale_factor: bool = False,
    ):

        if fit_var_scale_factor:
            self.calibrate_with_scale_factor(
                cal_loader=cal_loader,
                optimizer=partial(
                    torch.optim.LBFGS,
                    lr=0.01,
                    max_iter=100,
                    history_size=1000,
                    line_search_fn='strong_wolfe',
                ),
                scheduler=partial(
                    torch.optim.lr_scheduler.ConstantLR,
                    factor=1,
                ),
                num_epochs=1000,
            )

        # Freeze all parameters except variance head
        grad_flags = {}
        for name, param in self.model.named_parameters():
            grad_flags[name] = param.requires_grad
            if 'variance_head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

            # We don't fit the scale factor (we fit a whole model instead)
            if 'var_scale_factor' in name:
                param.requires_grad = False

        def loss(mean, var, target, reduction):
            return gaussian_nll_loss(
                mean, target, var=var, full=True, reduction=reduction,
            )

        RegressionBase.fit(
            self,
            train_loader=cal_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            patience=patience,
            min_epochs=min_epochs,
            es_criterion='gaussian_nll',
            alt_loss=loss,
            disable_train=True,
        )

        # Reset gradient flags
        for name, param in self.model.named_parameters():
            if grad_flags[name] is True:
                param.requires_grad = True
            else:
                param.requires_grad = False


    def calibrate(
        self,
        cal_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        num_epochs: int = 1000,
    ):
        self.calibrate_with_scale_factor(
            cal_loader=cal_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs
        )


    def predict(
        self,
        dataloader: DataLoader,
        use_var_scale_factor: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts with the MAP solution.

        Args:
            dataloader: A PyTorch dataloader containing the data to predict.

        Returns:
            A tuple of tensors with predictions (mean and variance)
            for the data and the corresponding targets.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        means = list()
        variances = list()
        targets = list()

        self.model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                if hasattr(x, 'features') and hasattr(x, 'mean'):
                    mean, var = self.model(
                        x, use_var_scale_factor=use_var_scale_factor, mean=x.mean, features=x.features,
                    )
                else:
                    mean, var = self.model(x, use_var_scale_factor=use_var_scale_factor)

                means.append(mean.detach().cpu())           # batch_size
                variances.append(var.detach().cpu())        # batch_size
                targets.append(y.detach().cpu())            # batch_size

        means = torch.concat(means, dim=0)                  # num_datapoints
        variances = torch.concat(variances, dim=0)          # num_datapoints
        targets = torch.concat(targets, dim=0)              # num_datapoints

        return means, variances, targets


    def compute_regression_stats(
        self,
        mean: torch.Tensor,
        var: Union[torch.Tensor, None],
        target: torch.Tensor,
    ) -> Dict:
        """
        Compute evaluation statistics for regression.

        Args:
            mean: A tensor with shape (num_datapoints,) with prediction means
            targets: A tensor with shape (num_datapoints,) with prediction
                targets.
            var: A tensor with shape (num_datapoints,) with prediction variances

        Returns:
            A dictionary with the evaluation statistics.
        """
        eval_stats = dict()
        mean = mean.detach().to('cpu')
        var = var.detach().to('cpu')
        target = target.detach().to('cpu').squeeze()

        # Metric computations assumes an output tensor of shape (S, N, K)
        # We have S = 1 (one model) and K = 2 (mean and variance, with constant
        # variance in homoscedastic setting)
        if mean.shape == var.shape:
            output = torch.stack([mean, var], dim=1).unsqueeze(0)           # S x N x K
            S, N, K = output.shape
        elif mean.dim() == 1 and var.dim() == 2:
            # Ensemble of variance models
            output = transform_shapes_for_variance_ensemble(mean, var) # S x N x K
            S, N, K = output.shape
        else:
            raise ValueError(
                f'Unknow shapes. '
                f'Mean shape: {mean.shape}. '
                f'Variance shape: {var.shape}. '
                f'Target shape: {target.shape}'
            )

        model_means = output[:,:,0]                                     # S x N
        model_vars = output[:,:,1]                                      # S x N
        ensemble_mean = model_means.mean(dim=0)                         #     N
        ensemble_aleatoric_var = model_vars.mean(dim=0)                 #     N
        ensemble_epistemic_var = torch.mean((model_means - ensemble_mean)**2, dim=0) # N
        ensemble_total_var = ensemble_aleatoric_var + ensemble_epistemic_var # N

        # Expected log predictive density
        log_densities = dist.Normal(
            model_means, torch.sqrt(model_vars)
        ).log_prob(target)                                             # S x N
        eval_stats['lpd'] = (
            -N*torch.log(torch.tensor(S))
            + torch.logsumexp(log_densities, dim=0).sum()
        ).item() / N

        # Expected calibration error / Mean absolute calibration error
        eval_stats['ece'] = expected_calibration_error_regression(
            ensemble_mean,
            torch.sqrt(ensemble_total_var).numpy(),
            target.numpy(),
            num_bins=15
        )

        if mean.dim() == 1 and var.dim() == 2:
            eval_stats['ece_mixture'] = regression_ece_for_gaussian_mixture(
                model_means.numpy(),
                torch.sqrt(model_vars).numpy(),
                target.numpy(),
                probs=np.linspace(0, 1, 15 + 1)[1:] # Same as in above function (num_bins = 15)
            )

        # Uncertainty toolbox metrics
        uct_metrics = uct.get_all_metrics(
            ensemble_mean.numpy(),
            torch.sqrt(ensemble_total_var).numpy(),
            target.numpy(),
            verbose=False,
        )
        for key, d in uct_metrics.items():
            if key != 'adv_group_calibration':
                eval_stats.update(d)

        # Loss (used for monitoring convergence)
        eval_stats['loss'] = self.loss(ensemble_mean, ensemble_total_var, target, reduction='mean').item()
        eval_stats['gaussian_nll'] = gaussian_nll_loss(
            ensemble_mean, target, var=ensemble_total_var, full=True, reduction='mean'
        ).item()

        return eval_stats
