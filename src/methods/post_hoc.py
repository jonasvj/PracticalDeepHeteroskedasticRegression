import torch
from copy import deepcopy
from torch.optim import Optimizer
from src.methods import RegressionBase
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.functional import gaussian_nll_loss
from src.data.qm9_utils import EfficientDataLoader
from src.data.qm9_utils import DataLoader as GraphDataLoader
from typing import Literal, Any, Callable, Optional


def apply_to_features(features, func: Callable, *args, **kwargs):
    if isinstance(features, torch.Tensor):
        return func(features, *args, **kwargs)
    elif isinstance(features, dict):
        return {k: apply_to_features(f, func, *args, **kwargs) for k, f in features.items()}
    elif isinstance(features, (list, tuple)):
        list_ = [apply_to_features(f, func, *args, **kwargs) for f in features]
        return tuple(list_) if isinstance(features, tuple) else list_
    else:
        raise ValueError(f'apply_to_features not implented for {type(features)}')


def get_features(features: Any, idx_vector: torch.Tensor, idx: int):
    def func(features, idx_vector, idx):
        features = features.detach()
        return features[idx_vector == idx]
    return apply_to_features(features, func, idx_vector, idx)


def create_feature_dataset(dataloader, model, device: str = 'cuda'):
    # Construct dataset with extracted features
    dataset = list()
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            mean, features = model.feature_extractor(x)

            for idx in range(x.num_graphs):
                x_i = x.get_example(idx)
                mean_i = mean[idx]
                features_i = get_features(features, x.batch, idx)

                x_i.update({'features': features_i, 'mean': mean_i})
                x_i = x_i.to('cpu')
                dataset.append(x_i)

    return dataset


class PostHocRegression(RegressionBase):


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


    def fit(*args, **kwargs):
        raise ValueError(
            'This is an post-hoc method and fit should not be called.'
        )


    def calibrate(
        self,
        cal_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        num_epochs: int = 1000,
        patience: int = 1000,
        min_epochs: int = 1000,
        precompute_features: bool = True,
        fit_var_scale_factor: bool = False,
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Construct dataset with extracted features
        if precompute_features:
            example_data = cal_loader.dataset[0]
            if hasattr(example_data, 'features') and hasattr(example_data, 'mean'):
                cal_dataset = cal_loader.dataset
                val_dataset = val_loader.dataset
            else:
                cal_dataset = create_feature_dataset(cal_loader, self.model, device)
                val_dataset = create_feature_dataset(val_loader, self.model, device)

            cal_loader = EfficientDataLoader(
                cal_dataset,
                batch_size=cal_loader.batch_size,
                shuffle=True,
                exclude_keys=['edge_index', 'edge_attr', 'smiles', 'name', 'idx'],
                num_workers=0,
                storage_device=device
            )
            val_loader = EfficientDataLoader(
                val_dataset,
                batch_size=val_loader.batch_size,
                shuffle=False,
                exclude_keys=['edge_index', 'edge_attr', 'smiles', 'name', 'idx'],
                num_workers=0,
                storage_device=device,
            )

        self.calibrate_with_variance_head(
            cal_loader=cal_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            patience=patience,
            min_epochs=min_epochs,
            fit_var_scale_factor=fit_var_scale_factor,
        )


    def calibrate_with_earlystopping(
        self,
        cal_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        num_epochs: int = 1000,
        patience: int = 1000,
        min_epochs: int = 1000,
        precompute_features: bool = True,
        train_val_split_ratio: Optional[float] = 0.8,
        split_seed: int = 0,
        fit_var_scale_factor: bool = False,
    ):
        # Split data into Calibration train and Calibration val
        cal_dataset = deepcopy(cal_loader.dataset)
        rng = torch.Generator(device='cpu').manual_seed(split_seed)
        perm = torch.randperm(len(cal_dataset), generator=rng)
        split_idx = int(train_val_split_ratio * len(cal_dataset))
        cal_train_dataset = cal_dataset[perm[:split_idx]]
        cal_val_dataset = cal_dataset[perm[split_idx:]]

        cal_train_loader = GraphDataLoader(
            cal_train_dataset,
            batch_size=cal_loader.batch_size,
            shuffle=True,
            num_workers=0,
        )
        cal_val_loader = GraphDataLoader(
            cal_val_dataset,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=0
        )

        # Save original variance head weights
        orig_var_head_weights = deepcopy(self.model.variance_head.state_dict())

        # Calibrate
        self.calibrate(
            cal_loader=cal_train_loader,
            val_loader=cal_val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            patience=patience,
            min_epochs=min_epochs,
            precompute_features=precompute_features,
            fit_var_scale_factor=fit_var_scale_factor,
        )

        # Get optimal number of steps and set max number of steps as current
        # step + number of optimal steps
        self.max_num_steps = deepcopy(self.step) + deepcopy(self.best_step)

        # Save scale factor if fitted and decrease max number of steps by steps
        # used for fitting scale factor (1000)
        if fit_var_scale_factor:
            scale_factor = deepcopy(self.model.variance_head.var_scale_factor)
            self.max_num_steps = self.max_num_steps - 1000

        # Reinitialize variance head
        self.model.variance_head.load_state_dict(orig_var_head_weights)
        if fit_var_scale_factor:
            self.model.variance_head.var_scale_factor.copy_(scale_factor)

        self.calibrate(
            cal_loader=cal_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            patience=None,
            min_epochs=None,
            precompute_features=precompute_features,
            fit_var_scale_factor=False,
        )
