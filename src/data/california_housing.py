import os
import torch
import numpy as np
import pytorch_lightning as pl
from typing import Optional, List, Union
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset


class CaliforniaHousingDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str = 'data/california_housing',
        batch_size_train: int = 256,
        batch_size_inference: int = 1024,
        num_workers: int = 0,
        splits: Union[List[int], List[float]] = [0.8, 0.1, 0.1],
        seed: int = 0,
        subset_size: Optional[int] = None,
        cal_subset_size: Optional[Union[int, float]] = None,
        cal_subset_seed: int = 0,
        normalization: str = 'quantile',
        name: str = 'california_housing',
        target: str = 'california_housing',
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size_train = batch_size_train
        self.batch_size_inference = batch_size_inference
        self.num_workers = num_workers
        self.splits = splits
        self.seed = seed
        self.subset_size = subset_size
        self.cal_subset_size = cal_subset_size
        self.cal_subset_seed = cal_subset_seed
        self.normalization = normalization
        self.name = name

        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.y_mean = None
        self.y_std = None

    def prepare_data(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        cache_path = os.path.join(self.data_dir, 'data.npz')
        if not os.path.exists(cache_path):
            data = fetch_california_housing()
            np.savez(cache_path, X=data.data, y=data.target)

    def setup(self, stage: Optional[str] = None) -> None:
        cache_path = os.path.join(self.data_dir, 'data.npz')
        loaded = np.load(cache_path)
        X, y = loaded['X'].astype(np.float32), loaded['y'].astype(np.float32)

        # Shuffle
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]

        if self.subset_size is not None:
            X, y = X[:self.subset_size], y[:self.subset_size]

        # Split
        n = len(X)
        if all(isinstance(s, float) for s in self.splits):
            split_sizes = [int(n * p) for p in self.splits]
        else:
            split_sizes = self.splits
        idx = np.cumsum(split_sizes).astype(int)

        X_train, y_train = X[:idx[0]], y[:idx[0]]
        X_val, y_val = X[idx[0]:idx[1]], y[idx[0]:idx[1]]
        X_test, y_test = X[idx[1]:], y[idx[1]:]

        # Feature normalization (fit on train only)
        if self.normalization == 'quantile':
            qt = QuantileTransformer(
                output_distribution='normal', random_state=self.seed
            )
            X_train = qt.fit_transform(X_train)
            X_val = qt.transform(X_val)
            X_test = qt.transform(X_test)

        # Target normalization
        self.y_mean = float(y_train.mean())
        self.y_std = float(y_train.std())

        self.data_train = TensorDataset(
            torch.from_numpy(X_train), torch.tensor(y_train),
        )
        self.data_val = TensorDataset(
            torch.from_numpy(X_val), torch.tensor(y_val),
        )
        self.data_test = TensorDataset(
            torch.from_numpy(X_test), torch.tensor(y_test),
        )

        # Calibration data = validation by default
        self.data_cal = self.data_val
        if self.cal_subset_size is not None:
            num_cal = (
                self.cal_subset_size if self.cal_subset_size > 1
                else int(self.cal_subset_size * len(self.data_cal))
            )
            rng_cal = np.random.default_rng(self.cal_subset_seed)
            order = rng_cal.permutation(len(self.data_cal))[:int(num_cal)]
            self.data_cal = torch.utils.data.Subset(self.data_cal, order)

    def get_target_stats(self):
        return torch.tensor(self.y_mean), torch.tensor(self.y_std)

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def cal_dataloader(self, shuffle=True):
        return DataLoader(
            self.data_cal,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
        )
