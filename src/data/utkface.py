import os
import torch
import subprocess
import numpy as np
import torchvision
from PIL import Image
import pytorch_lightning as pl
from typing import Literal, Optional, List, Union
from torch.utils.data import Dataset, DataLoader


def download_utkface(root: str, force: bool = False) -> None:
    """Download and extract the UTKFace dataset into ``root``.

    The three archives are downloaded from Google Drive with ``gdown``.
    If ``root`` already exists and contains PNG/JPG files this function does
    nothing unless ``force`` is ``True``.
    """
    if os.path.isdir(root) and any(
        fname.lower().endswith(('.jpg', '.png')) for fname in os.listdir(root)
    ) and not force:
        return

    os.makedirs(root, exist_ok=True)
    gdrive_ids = [
        '1mb5Z24TsnKI3ygNIlX6ZFiwUj0_PmpAW',
        '19vdaXVRtkP-nyxz1MYwXiFsh_m_OL72b',
        '1oj9ZWsLV2-k2idoW_nRSrLQLUP3hus3b',
    ]
    gdrive_fnames = [
        'part1.tar.gz',
        'part2.tar.gz',
        'part3.tar.gz',
    ]

    for id, fname in zip(gdrive_ids, gdrive_fnames):
        subprocess.run(['gdown', '-O', os.path.join(root, fname), id], check=True)
        subprocess.run(['tar', '-xzf', os.path.join(root, fname), '--strip-components=1', '-C', root], check=True)


class UTKFaceDataset(Dataset):
    def __init__(
        self,
        root: str,
        target: Literal['age'] = 'age',
        transform=None,
    ):
        self.root = root
        self.transform = transform
        self.target = target
        self.files = sorted([
            f for f in os.listdir(root)
            if f.lower().endswith(('.jpg', '.png'))
        ])

        if self.target != 'age':
            raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        age, _ = name.split('_', maxsplit=1)
        age = float(age)
        img = Image.open(os.path.join(self.root, name)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, age


class UTKFaceDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        target: str = 'age',
        batch_size_train: int = 32,
        batch_size_inference: int = 32,
        num_workers: int = 0,
        splits: Union[List[int], List[float]] = [0.8, 0.1, 0.1],
        seed: int = 0,
        subset_size: Optional[int] = None,
        cal_subset_size: Optional[Union[int, float]] = None,
        cal_subset_seed: int = 0,
        name: str = 'utkface',
    ):
        super().__init__()
        self.data_dir = data_dir
        self.target = target
        self.batch_size_train = batch_size_train
        self.batch_size_inference = batch_size_inference
        self.num_workers = num_workers
        self.splits = splits
        self.seed = seed
        self.subset_size = subset_size
        self.cal_subset_size = cal_subset_size
        self.cal_subset_seed = cal_subset_seed

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self, force_reload: bool = False) -> None:
        # ensure the images are present; download if necessary
        if not os.path.isdir(self.data_dir) or force_reload:
            print('Downloading UTKFace dataset...')
            download_utkface(self.data_dir, force=force_reload)

        if not os.path.isdir(self.data_dir) or not any(
            fname.lower().endswith(('.jpg', '.png'))
            for fname in os.listdir(self.data_dir)
        ):
            raise RuntimeError(f"UTKFace directory {self.data_dir} not found or empty")

    def setup(self, stage: Optional[str] = None) -> None:
        transform = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
        dataset = UTKFaceDataset(
            self.data_dir,
            target=self.target,
            transform=transform,
        )

        # shuffle
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(dataset))
        dataset = [dataset[i] for i in perm]          # list is fine for slicing

        if self.subset_size is not None:
            dataset = dataset[:self.subset_size]

        # split
        if all(isinstance(x, int) for x in self.splits):
            split_sizes = self.splits
        else:
            split_sizes = [int(len(dataset) * p) for p in self.splits]

        idx = np.cumsum(split_sizes).astype(int)
        self.data_train = dataset[:idx[0]]
        self.data_val   = dataset[idx[0]:idx[1]]
        self.data_test  = dataset[idx[1]:]

        # calibration = validation by default
        self.data_cal = self.data_val
        if self.cal_subset_size is not None:
            num_cal = (self.cal_subset_size if self.cal_subset_size > 1
                       else int(self.cal_subset_size * len(self.data_cal)))
            rng = np.random.default_rng(self.cal_subset_seed)
            order = rng.permutation(len(self.data_cal))[:num_cal]
            self.data_cal = [self.data_cal[i] for i in order]


    def get_target_stats(self):
        ys = list()
        for x, y in self.train_dataloader(shuffle=False):
            ys.append(y.detach().cpu())

        y = torch.cat(ys, dim=0)
        return y.mean(), y.std()


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
