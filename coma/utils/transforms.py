from typing import List, Tuple, Type

import numpy as np
import pyvista as pv
import torch
import torchvision.transforms as T

from coma.utils import registration as reg
from coma.datasets import get_data_from_polydata


class GetVerticesFromPolyData:

    def __call__(self, polydata: pv.PolyData) -> np.ndarray:
        data = get_data_from_polydata(polydata)
        return data.x


class RigidRegistrationTransform:

    def __init__(self, registration_obj: Type[reg.RegistrationBase]):
        # TODO: Maybe remeshing parameters
        self.reg_obj = registration_obj

    def __call__(self, image_vertices: np.ndarray) -> np.ndarray:
        if not hasattr(self, 'reg'):
            # TODO: Maybe perform remeshing here 
            self.reg = self.reg_obj(image_vertices)
            return image_vertices
        return self.reg.align(image_vertices)


def get_transforms():
    return T.Compose([
        GetVerticesFromPolyData(),
        lambda x: x.cpu().detach().numpy(),
        RigidRegistrationTransform(reg.RigidRegistration),
        T.ToTensor(),
    ])
