import os
import zipfile
import urllib.request
import numpy as np
import torch
import pytorch_lightning as pl
from typing import Optional, List, Union
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset


_YEAR_URL = (
    'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/'
    'YearPredictionMSD.txt.zip'
)


class YearDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str = 'data/year',
        batch_size_train: int = 1024,
        batch_size_inference: int = 1024,
        num_workers: int = 0,
        splits: Union[List[int], List[float]] = [0.9, 0.1],
        seed: int = 0,
        subset_size: Optional[int] = None,
        cal_subset_size: Optional[Union[int, float]] = None,
        cal_subset_seed: int = 0,
        normalization: str = 'quantile',
        name: str = 'year',
        target: str = 'year',
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
        if os.path.exists(cache_path):
            return

        zip_path = os.path.join(self.data_dir, 'YearPredictionMSD.txt.zip')
        txt_path = os.path.join(self.data_dir, 'YearPredictionMSD.txt')

        if not os.path.exists(txt_path):
            print(f'Downloading Year dataset from {_YEAR_URL} ...')
            urllib.request.urlretrieve(_YEAR_URL, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(self.data_dir)

        print('Loading Year dataset ...')
        raw = np.loadtxt(txt_path, delimiter=',', dtype=np.float32)
        y = raw[:, 0]
        X = raw[:, 1:]
        np.savez(cache_path, X=X, y=y)
        # Clean up intermediate files
        for p in (zip_path, txt_path):
            if os.path.exists(p):
                os.remove(p)

    def setup(self, stage: Optional[str] = None) -> None:
        cache_path = os.path.join(self.data_dir, 'data.npz')
        loaded = np.load(cache_path)
        X, y = loaded['X'], loaded['y']

        # Standard split for Year: first 463715 for train+val, rest for test
        n_trainval = 463715
        X_trainval, y_trainval = X[:n_trainval], y[:n_trainval]
        X_test, y_test = X[n_trainval:], y[n_trainval:]

        # Shuffle train+val
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n_trainval)
        X_trainval, y_trainval = X_trainval[perm], y_trainval[perm]

        if self.subset_size is not None:
            X_trainval = X_trainval[:self.subset_size]
            y_trainval = y_trainval[:self.subset_size]

        # Split train/val
        if all(isinstance(s, float) for s in self.splits):
            n_train = int(len(X_trainval) * self.splits[0])
        else:
            n_train = self.splits[0]

        X_train, y_train = X_trainval[:n_train], y_trainval[:n_train]
        X_val, y_val = X_trainval[n_train:], y_trainval[n_train:]

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
