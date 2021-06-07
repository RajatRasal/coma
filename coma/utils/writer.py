import os
import time
import json
from glob import glob

import pyvista as pv
import torch
from torch.utils.tensorboard import SummaryWriter


class MeshWriter:
    def __init__(self, args, template: pv.PolyData):
        self.args = args
        self.template = template

        self.faces = torch.tensor(self.template.faces.reshape(-1, 4)[:, 1:])

        out_dir = self.args.out_dir

        if os.path.lexists(out_dir):
            versions = os.listdir(out_dir)
            max_version = max(
                int(version.split('_')[-1])
                for version in versions
            )
            self.exp_dir = f'{out_dir}/version_{max_version + 1}'
        else:
            os.mkdir(out_dir)
            self.exp_dir = f'{out_dir}/version_0'
        
        os.mkdir(self.exp_dir)

        self.writer = SummaryWriter(self.exp_dir)
        hparams = {k: str(v) for k, v in vars(self.args).items()}
        self.writer.add_hparams(
            hparam_dict=hparams,
            metric_dict={'hpmetrics': 0.1},
        )

        with open(f'{self.exp_dir}/hparam.json', 'w') as f:
            json.dump(hparams, f)

    def write_scalars(self, epoch, train=True, **scalars):
        section = 'Train' if train else 'Val'
        for k, v in scalars.items():
            print(k, v)
            self.writer.add_scalar(f'{section}/{k}', v, epoch)

    # def write_meshes(self, epoch, verts, train=False):
    #     for i in range(len(verts)):
    #         self.write_mesh(epoch, verts[i], train)

    def write_meshes(self, epoch, verts, train=False):
        """
        Vertices should correspond to faces provided in the constructor
        """
        assert verts.shape[1] - 1 == int(self.faces.max())

        section = 'Train' if train else 'Val'
        camera_config = {
	    'cls': 'PerspectiveCamera',
            'fov': 75,
            'aspect': 0.9,
        }
        material_config = {
            'cls': 'MeshDepthMaterial',
            'wireframe': True,
        }
        config_dict = {
            'material': material_config,
            'camera': camera_config,
        }
        print(self.faces.unsqueeze(0).shape)

        self.writer.add_mesh(
            section,
            vertices=verts,
            colors=None,
            faces=self.faces.unsqueeze(0).repeat((verts.shape[0], 1, 1)),
            global_step=epoch,
            config_dict=config_dict,
        )

    def save_model_checkpoint(self, model, epoch):
        path = os.path.join(self.exp_dir, 'checkpoint.pt')
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, path)

    # def save_checkpoint(self, model, optimizer, scheduler, epoch):
    #     torch.save(
    #         {
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'scheduler_state_dict': scheduler.state_dict(),
    #         },
    #         os.path.join(
    #             self.args.checkpoints_dir,
    #             'checkpoint_{:03d}.pt'.format(epoch))
    #         )
