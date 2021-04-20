import os
from collections import defaultdict, namedtuple
from glob import glob
from typing import List

import pyvista as pv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Compose


BatchWrapper = namedtuple('BatchWrapper', ['x'])


class UKBBMeshDataset(Dataset):
    """
    Path should have the following directory structure:
    /path
    ├── 1000000
    │   ├── T1_first-BrStem_first.vtk
    │   ├── T1_first-L_Accu_first.vtk
    │   ├── T1_first-L_Amyg_first.vtk 
    │   ├── T1_first-L_Caud_first.vtk
    │   ├── T1_first-L_Hipp_first.vtk
    │   ├── T1_first-L_Pall_first.vtk
    │   ├── T1_first-L_Puta_first.vtk
    │   ├── T1_first-L_Thal_first.vtk
    │   ├── T1_first-R_Accu_first.vtk
    │   ├── T1_first-R_Amyg_first.vtk
    │   ├── T1_first-R_Caud_first.vtk
    │   ├── T1_first-R_Hipp_first.vtk
    │   ├── T1_first-R_Pall_first.vtk
    │   ├── T1_first-R_Puta_first.vtk
    │   └── T1_first-R_Thal_first.vtk
    ├── 1000001
    │   ├── T1_first-BrStem_first.vtk
        ...
    ...

    We expect that there should be 7 meshes for the left and right side of the
    brain and another mesh for the brain stem.
    """
    def __init__(self, path: str, substructures: List[str], split: float = 0.8,
        train: bool = True, transform: Compose = None,
    ):
        # TODO: Reload option - if not specified, then just use cached info file is available
        super().__init__()
        self.path = path
        self.substructures = sorted(substructures)
        self.transform = transform
        self.data_subject_ids = []
        self.lookup_dict = defaultdict(lambda: defaultdict(str))
        self.flat_list = []
        self.split = split
        self.train = train
        self.__read_path_structure()

    def __read_path_structure(self):
        # Find all numbers, sort them in ascending order
        # Form dictionary {no: {substr: full_path}}
        # Form flat list [fullpath]
        data_subject_ids = sorted(
            [int(x) for x in os.listdir(self.path) if x.isdigit()]
        )

        split_idx = int(len(data_subject_ids) * self.split)

        if self.train:
            self.data_subject_ids = data_subject_ids[:split_idx]
        else:
            self.data_subject_ids = data_subject_ids[split_idx:]

        for _id in self.data_subject_ids:
            for substructure in self.substructures:
                full_path = f'{self.path}/{_id}/T1_first-{substructure}_first.vtk'
                if not os.path.exists(full_path):
                    continue
                self.lookup_dict[_id][substructure] = full_path
                self.flat_list.append(full_path)

    def get_data_subject_ids(self):
        return self.data_subject_ids

    def lookup_mesh(self, data_subject_id: int, substructure: str) -> str:
        res = self.lookup_dict[data_subject_id]
        
        if res == {}:
            return None

        res = res[substructure]

        return None if res == {} else res

    def __load_mesh_file(self, full_path: str):
        return pv.PolyData(full_path)

    def get_mesh_by_lookup(self, data_subject_id: int, substructure: str):
        assert substructure in self.substructures

        full_path = self.lookup_mesh(data_subject_id, substructure)

        return None if full_path is None else self.__load_mesh_file(full_path)

    def __len__(self):
        return len(self.flat_list)

    def get_raw(self, index):
        return self.__load_mesh_file(self.flat_list[index])

    def __getitem__(self, index):
        mesh = self.__load_mesh_file(self.flat_list[index])
        return self.transform(mesh) if self.transform else mesh


def get_data_from_polydata(polydata):
    faces = polydata.faces.reshape(-1, 4)[:, 1:].T
    normal = polydata.point_normals
    pos = polydata.points
    edge_index = np.hstack((faces[0:2], faces[[0, 2]], faces[1:]))
    data = Data(
        x=torch.tensor(pos).double(),
        edge_index=torch.tensor(edge_index).long(),
        pos=torch.tensor(pos),
        normal=torch.tensor(normal),
        face=torch.tensor(faces),
    )
    return data


class VerticesDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):

        def collate_fn_for_ukbb_meshes_pipeline(data_list: List[torch.Tensor]) -> torch.Tensor:
            batch = torch.vstack([data.double() for data in data_list])
            return BatchWrapper(x=batch)

        super(VerticesDataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=collate_fn_for_ukbb_meshes_pipeline,
            **kwargs,
        )
