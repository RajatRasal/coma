import os
from collections import defaultdict
from glob import glob
from typing import List

import pyvista as pv
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch_geometric.data import Data, Batch
from tqdm import tqdm


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
    def __init__(self, path: str, substructures: List[str]):
        super().__init__()
        self.path = path
        self.substructures = sorted(substructures)
        # Read the folders in the path
        # Sort them in order
        self.data_subject_ids = []
        self.lookup_dict = defaultdict(lambda: defaultdict(str))
        self.flat_list = []
        self.__read_path_structure()

    def __read_path_structure(self):
        # Find all numbers, sort them in ascending order
        # Form dictionary {no: {substr: full_path}}
        # Form flat list [fullpath]
        self.data_subject_ids = sorted(
            [int(x) for x in os.listdir(self.path) if x.isdigit()]
        )

        for _id in self.data_subject_ids:
            for substructure in self.substructures:
                full_path = f'{self.path}/{_id}/T1_first-{substructure}_first.vtk'
                if not os.path.exists(full_path):
                    continue
                self.lookup_dict[_id][substructure] = full_path
                self.flat_list.append(full_path)

    def get_data_subject_ids(self):
        return self.mesh_ids

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

    def __getitem__(self, index):
        return self.__load_mesh_file(self.flat_list[index])


class PolyDataDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):

        def collate_polydata(polydata_list):
            data_list = []
            for polydata in polydata_list:
                face = polydata.faces.reshape(-1, 4)[:, 1:].T
                normal = polydata.point_normals
                pos = polydata.points
                edge_index = np.hstack((faces[0:2], faces[[0, 2]], faces[1:]))
                data = Data(
                    edge_index=torch.tensor(edge_index).long(),
                    pos=torch.tensor(pos),
                    normal=torch.tensor(normal),
                    face=torch.tensor(face)
                )
                data_list.append(data)
            batch = Batch.from_data_list(data_list)
            return batch

        super(PolyDataDataLoader, self).__init__(
            dataset, batch_size, shuffle, collate_fn=collate_polydata
        )


if __name__ == '__main__':
    path = '/vol/biomedic3/bglocker/brainshapes/'
    import pickle
    dataset = UKBBMeshDataset(path, ['R_Thal'])
    print(dataset[0])
    pickle.dump(dataset, '/tmp/dataset')
