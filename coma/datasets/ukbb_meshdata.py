import os
import pandas as pd
import pickle
from collections import defaultdict, namedtuple
from glob import glob
from typing import List, Dict, Tuple

import pyvista as pv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Compose


BatchWrapper = namedtuple('BatchWrapper', ['x', 'features'])
# NOTE: Subdict function needed to be able to pickle the defaultdict
def _subdict(): return defaultdict(str)


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

    def __init__(self, path: str, substructures: List[str],
        features_df: pd.DataFrame, feature_name_map: Dict[str, str],
        split: float = 0.8, data_subject_id_colname: str = 'eid',
        train: bool = True, transform: Compose = None, reload_path: bool = True,
        cache_path: str = '.'
    ):
        # TODO: Reload option - if not specified, then just use cached info file is available
        super().__init__()
        self.path = path
        self.cache_path = cache_path
        self.substructures = sorted(substructures)
        self.transform = transform
        self.data_subject_ids = []
        self.lookup_dict = defaultdict(_subdict)
        self.flat_list = []
        self.split = split
        self.train = train
        self.reload_path = reload_path

        self.feature_name_map = feature_name_map
        self.data_subject_id_colname = data_subject_id_colname
        cols = [self.data_subject_id_colname] + list(self.feature_name_map.keys())
        self.features_df = features_df[cols]
        self.features_df_renamed = self.features_df.rename(
            columns=self.feature_name_map
        )

        pickle_files = '_'.join(substructures)
        self.data_sub_file = f'{self.cache_path}/data_subject_ids_{pickle_files}.pickle'
        self.lookup_dict_file = f'{self.cache_path}/lookup_dict_{pickle_files}.pickle' 
        self.flat_list_file = f'{self.cache_path}/flat_list_{pickle_files}.pickle' 
        self.__read_path_structure()

    def __read_path_structure(self):
        # Find all numbers, sort them in ascending order
        # Form dictionary {no: {substr: full_path}}
        # Form flat list [fullpath]
        if not self.reload_path:
            with open(self.data_sub_file, 'rb') as data_sub_file, \
                open(self.lookup_dict_file, 'rb') as lookup_dict_file, \
                open(self.flat_list_file, 'rb') as flat_list_file:
                self.data_subject_ids = pickle.load(data_sub_file)
                self.lookup_dict = pickle.load(lookup_dict_file)
                self.flat_list = pickle.load(flat_list_file)
                return

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
                full_path = self.create_vtk_path(self.path, _id, substructure)
                if not os.path.exists(full_path):
                    continue
                self.lookup_dict[_id][substructure] = full_path
                self.flat_list.append(full_path)

        with open(self.data_sub_file, 'wb') as data_sub_file, \
            open(self.lookup_dict_file, 'wb') as lookup_dict_file, \
            open(self.flat_list_file, 'wb') as flat_list_file:
            pickle.dump(self.data_subject_ids, data_sub_file)
            pickle.dump(self.lookup_dict, lookup_dict_file)
            pickle.dump(self.flat_list, flat_list_file)

    def get_data_subject_ids(self):
        return self.data_subject_ids

    def lookup_mesh(self, data_subject_id: int, substructure: str) -> str:
        res = self.lookup_dict[data_subject_id]
        
        if res == {}:
            return None

        res = res[substructure]

        return None if res == {} else res

    def __load_mesh_file(self, full_path: str):
        return pv.read(full_path)

    def lookup_features(self, data_subject_id: int):
        mask = self.features_df_renamed[self.data_subject_id_colname] == data_subject_id
        features = self.features_df_renamed.loc[mask].head(1)
        return features

    def get_mesh_by_lookup(self, data_subject_id: int, substructure: str):
        assert substructure in self.substructures

        full_path = self.lookup_mesh(data_subject_id, substructure)
        features = self.lookup_features(data_subject_id)

        return None if full_path is None else self.__load_mesh_file(full_path), features

    def __len__(self):
        return len(self.flat_list)

    def create_vtk_path(self, path, _id, substructure):
        if path[-1] == '/':
            path = path[:-1]
        return f'{path}/{_id}/T1_first-{substructure}_first.vtk'

    def get_id_from_vtk_path(self, vtk_path):
        base_path_splits = len(self.path.split('/'))
        return vtk_path.split('/')[base_path_splits] 

    def get_raw(self, index):
        return self.__load_mesh_file(self.flat_list[index])

    def __getitem__(self, index):
        path = self.flat_list[index]
        mesh = self.__load_mesh_file(path)
        data_subject_id = self.get_id_from_vtk_path(path)
        features = self.lookup_features(int(data_subject_id))
        mesh = self.transform(mesh) if self.transform else mesh
        return mesh, features


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

        def collate_fn_for_ukbb_meshes_pipeline(data_list: List[Tuple[torch.Tensor, Tuple[int, ...]]]) -> torch.Tensor:
            batch = torch.vstack([data[0].double() for data in data_list])
            features = pd.concat([data[1] for data in data_list])
            return BatchWrapper(x=batch, features=features)

        super(VerticesDataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=collate_fn_for_ukbb_meshes_pipeline,
            **kwargs,
        )
