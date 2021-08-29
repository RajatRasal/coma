from typing import Optional, Callable, List

import shutil
import os.path as osp

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset, Data, extract_zip
from torch_geometric.io import read_ply


class FAUST(InMemoryDataset):
    """
    Adapted from Pytorch Geometric FAUST dataloader
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/faust.html#FAUST
    """

    url = 'http://faust.is.tue.mpg.de/'

    def __init__(self, root: str, train: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> str:
        return 'MPI-FAUST.zip'

    @property
    def processed_file_names(self) -> List[str]:
        return ['training.pt', 'test.pt']

    def download(self):
        raise RuntimeError(
            f"Dataset not found. Please download '{self.raw_file_names}' from "
            f"'{self.url}' and move it to '{self.raw_dir}'")

    def process(self):
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)

        path = osp.join(self.raw_dir, 'MPI-FAUST', 'training', 'registrations')
        path = osp.join(path, 'tr_reg_{0:03d}.ply')
        data_list = []
        for i in range(100):
            data = read_ply(path.format(i))
            data.y = torch.tensor([i % 10], dtype=torch.long)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list[:80]), self.processed_paths[0])
        torch.save(self.collate(data_list[80:]), self.processed_paths[1])
        shutil.rmtree(osp.join(self.raw_dir, 'MPI-FAUST'))

# TODO: Refactor BatchWrapper
from collections import namedtuple
BatchWrapper = namedtuple('BatchWrapper', ['x', 'features'])

class FAUSTDataLoader(DataLoader):

    def __init__(self, dataset: FAUST, batch_size=1, shuffle=False, **kwargs):

        def collate_fn(data_list: List[Data]):
            batch = torch.vstack([data.pos for data in data_list])
            batch = batch.reshape(-1, *data_list[0].pos.shape).double()
            return BatchWrapper(x=batch, features=[])

        super(FAUSTDataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=collate_fn,
            **kwargs,
        )
