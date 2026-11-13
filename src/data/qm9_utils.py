"""
Source:
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/loader/dataloader.py

Minor change to the Collater class such that the dataloader returns batches in
the form (batch, y) instead of (batch). 
"""
from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union, Literal

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Dataset, Data
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.typing import TensorFrame, torch_frame


class Collater:
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, BaseData):
            batch = Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
            batch.shape = torch.tensor([len(batch.y)], dtype=int)
            return (batch, batch.y)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, TensorFrame):
            return torch_frame.cat(batch, dim=0)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")


class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(dataset, follow_batch, exclude_keys),
            **kwargs,
        )

def get_features(features, gather_indices: torch.LongTensor):
    if isinstance(features, torch.Tensor):
        return features[gather_indices]
    elif isinstance(features, dict):
        return {k: get_features(f, gather_indices) for k, f in features.items()}
    elif isinstance(features, (list, tuple)):
        list_ = [get_features(f, gather_indices) for f in features]
        return tuple(list_) if isinstance(features, tuple) else list_
    else:
        raise ValueError(f'apply_to_features not implented for {type(features)}')


class EfficientCollater:
    def __init__(
        self,
        data_list: List[BaseData],
        storage_device: Literal['cpu', 'cuda'],
        exclude_keys: Optional[List[str]] = None,
    ):  
        # Dataset as a batch
        self.data_as_batch = Batch.from_data_list(
            data_list,
            exclude_keys=exclude_keys
        )
        self.data_as_batch = self.data_as_batch.to(storage_device)


    def __call__(self, idx_batch: List[int]) -> Any:
        if len(idx_batch) == self.data_as_batch.num_graphs:
            return (self.data_as_batch, self.data_as_batch.y)

        data = self.data_as_batch
        ptr = data.ptr
        device = ptr.device
        idx_batch = torch.tensor(idx_batch, device=ptr.device, dtype=ptr.dtype)

        # Number of nodes per selected graph
        offsets = ptr[idx_batch]
        sizes = ptr[idx_batch + 1] - offsets

        # Total nodes in batch
        total_nodes = sizes.sum()

        # Local node indices within each selected graph
        cumsum_sizes = torch.cumsum(sizes, 0)
        local_idx = (
            torch.arange(total_nodes, device=device)
            - torch.repeat_interleave(cumsum_sizes - sizes, sizes)
        )

        # Global node indices
        gather_indices = offsets.repeat_interleave(sizes) + local_idx # [num_nodes_in_batch]

        # Batch graph indices
        graph_indices = torch.repeat_interleave(
            torch.arange(len(idx_batch), device=device),
            sizes
        )

        # Pointer vector
        batch_ptr = torch.cat(
            [torch.tensor([0], device=device, dtype=cumsum_sizes.dtype), cumsum_sizes],
            dim=0
        )

        batch = Data(
            y=data.y[idx_batch],
            pos=data.pos[gather_indices],
            z=data.z[gather_indices],
            features=get_features(data.features, gather_indices),
            mean=data.mean[idx_batch],
            batch=graph_indices,
            num_graphs=len(idx_batch),
            ptr=batch_ptr
        )

        return (batch, batch.y)


class IdxDataset(torch.utils.data.Dataset):

    def __init__(self, size: int):
        self.size = size


    def __len__(self):
        return self.size
    

    def __getitem__(self, index):
        return index


    def __getitems__(self, indices: list):
        return indices


class EfficientDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Sequence[BaseData],
        storage_device: Literal['cpu', 'cuda'],
        batch_size: int = 1,
        shuffle: bool = False,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.exclude_keys = exclude_keys

        super().__init__(
            IdxDataset(len(dataset)),
            batch_size,
            shuffle,
            collate_fn=EfficientCollater(dataset, storage_device, exclude_keys),
            **kwargs,
        )