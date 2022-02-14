from typing import Union, List
from collections.abc import Mapping, Sequence
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, HeteroData, Dataset, Batch
import torch


class Collater(object):
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def collate(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return [None, None]
        elem = batch[0]
        if isinstance(elem, Data) or isinstance(elem, HeteroData):
            return Batch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('Invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)


class SafeDataLoader(torch.utils.data.DataLoader):
    def __init__(
            self,
            dataset: Union[Dataset, List[Data], List[HeteroData]],
            batch_size: int = 1,
            shuffle: bool = False,
            follow_batch: List[str] = [],
            exclude_keys: List[str] = [],
            **kwargs,
    ):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(dataset, batch_size, shuffle,
                         collate_fn=Collater(follow_batch,
                                             exclude_keys), **kwargs)
