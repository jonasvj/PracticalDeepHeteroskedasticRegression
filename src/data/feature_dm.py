import numpy as np
import pytorch_lightning as pl
from src.data.qm9_utils import DataLoader
from typing import Optional, List, Union, Literal
from src.data.feature_datasets import UMAFeatureDataset, AllScAIPFeatureDataset

class FeatureDataModule(pl.LightningDataModule):

    # Specify unit conversions (eV to meV).
    unit_conversion = {0: (lambda t: 1000*t)}

    def __init__(
        self,
        target: int = 0, # Unused but here for compability
        data_dir: str = 'data/omol25/val',
        batch_size_train: int = 32,
        batch_size_inference: int = 32,
        num_workers: int = 0,
        splits: Union[List[int]] = [10_000, 10_000], # Validation / calibration and test
        data_split: int = 0,
        name: Literal['uma', 'allscaip'] = 'uma',
        cal_subset_size: Optional[Union[int, float]] = None,
    ) -> None:
        super().__init__()
        self.target = target
        self.data_dir = data_dir
        self.batch_size_train = batch_size_train
        self.batch_size_inference = batch_size_inference
        self.num_workers = num_workers
        self.splits = splits
        self.data_split = data_split
        self.name = name
        self.cal_subset_size = cal_subset_size

        self.data_train = None
        self.data_val = None
        self.data_test = None


    def setup(self, stage: Optional[str] = None) -> None:
        if self.name == 'uma':
            dataset = UMAFeatureDataset(
                root=self.data_dir, split=self.data_split, force_reload=False
            )
        elif self.name == 'allscaip':
            dataset = AllScAIPFeatureDataset(
                root=self.data_dir, split=self.data_split, force_reload=False
            )

        split_idx = np.cumsum(self.splits)
        self.data_val = dataset[:split_idx[0]]
        self.data_test = dataset[split_idx[0]:split_idx[1]]

        # Calibration data (same as validation)
        self.data_cal = dataset[:split_idx[0]]

        if self.cal_subset_size is not None:
            num_cal = (
                self.cal_subset_size if self.cal_subset_size > 1 else
                int(self.cal_subset_size*len(self.data_cal))
            )
            self.data_cal = self.data_cal[:num_cal]


    def train_dataloader(self, shuffle=True) -> DataLoader:
        raise NotImplementedError(f'No training features for {self.name}')


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
