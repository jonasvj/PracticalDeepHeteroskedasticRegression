import torch
import numpy as np
import pytorch_lightning as pl
from .qm9_utils import DataLoader
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from typing import Optional, List, Union
from torch_geometric.transforms import BaseTransform


class GetTarget(BaseTransform):
    def __init__(self, target: Optional[int] = None) -> None:
        self.target = target

    def forward(self, data: Data) -> Data:
        if self.target is not None:
            data.y = data.y[:, self.target]
        return data


class QM9DataModule(pl.LightningDataModule):

    # Specify unit conversions (eV to meV).
    unit_conversion = {
        i: (lambda t: 1000*t) if i not in [0, 1, 5, 11, 16, 17, 18]
        else (lambda t: t)
        for i in range(19)
    }

    def __init__(
        self,
        target: int = 0,
        data_dir: str = 'data/',
        batch_size_train: int = 32,
        batch_size_inference: int = 32,
        num_workers: int = 0,
        splits: Union[List[int], List[float]] = [0.8, 0.1, 0.1],
        seed: int = 0,
        subset_size: Optional[int] = None,
        name: str = 'qm9',
        cal_subset_size: Optional[Union[int, float]] = None,
        cal_subset_seed: int = 0,
    ) -> None:
        super().__init__()
        self.target = target
        self.data_dir = data_dir
        self.batch_size_train = batch_size_train
        self.batch_size_inference = batch_size_inference
        self.num_workers = num_workers
        self.splits = splits
        self.seed = seed
        self.subset_size = subset_size
        self.name = name
        self.cal_subset_size = cal_subset_size
        self.cal_subset_seed = cal_subset_seed

        self.data_train = None
        self.data_val = None
        self.data_test = None


    def prepare_data(self, force_reload: bool = False) -> None:
        # Download data
        QM9(root=self.data_dir, force_reload=force_reload)


    def setup(self, stage: Optional[str] = None) -> None:
        dataset = QM9(root=self.data_dir, transform=GetTarget(self.target))

        # Shuffle dataset
        rng = np.random.default_rng(seed=self.seed)
        dataset = dataset[rng.permutation(len(dataset))]

        # Subset dataset
        if self.subset_size is not None:
            dataset = dataset[:self.subset_size]

        # Split dataset
        if all([type(split) == int for split in self.splits]):
            split_sizes = self.splits
        elif all([type(split) == float for split in self.splits]):
            split_sizes = [int(len(dataset) * prop) for prop in self.splits]

        split_idx = np.cumsum(split_sizes)
        self.data_train = dataset[:split_idx[0]]
        self.data_val = dataset[split_idx[0]:split_idx[1]]
        self.data_test = dataset[split_idx[1]:split_idx[2]]

        # Calibration data = validation data
        self.data_cal = self.data_val

        if self.cal_subset_size is not None:
            num_cal = (
                self.cal_subset_size if self.cal_subset_size > 1 else
                int(self.cal_subset_size*len(self.data_cal))
            )
            rng = np.random.default_rng(seed=self.cal_subset_seed)
            self.data_cal = self.data_cal[
                rng.permutation(len(self.data_cal))[:num_cal]
            ]


    def get_target_stats(self, remove_atom_refs=False, divide_by_atoms=False):
        atom_refs = self.data_train.atomref(self.target)

        ys = list()
        for batch, _ in self.train_dataloader(shuffle=False):
            y = batch.y.clone()
            if remove_atom_refs and atom_refs is not None:
                y.index_add_(
                    dim=0, index=batch.batch, source=-atom_refs[batch.z].squeeze(-1)
                )
            if divide_by_atoms:
                _, num_atoms  = torch.unique(batch.batch, return_counts=True)
                y = y / num_atoms
            ys.append(y)

        y = torch.cat(ys, dim=0)
        return y.mean(), y.std(), atom_refs


    def train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )


    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
        )


    def cal_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.data_cal,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )


    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
        )
