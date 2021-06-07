import torch
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import pandas as pd
from pyro.infer import SVI, Trace_ELBO
from torch.optim import Adam
import pyvista as pv
# from pyro.optim.lr_scheduler import PyroLRScheduler
from pyro.optim import StepLR

from coma.models import init_coma
from coma.models.elbo import CustomELBO
from coma.datasets.ukbb_meshdata import (
    UKBBMeshDataset, VerticesDataLoader, get_data_from_polydata
)
from coma.utils import transforms, writer
from coma.utils.train_eval_svi import train_eval_svi

import json


parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--out_dir', type=str, default='experiments')
parser.add_argument('--version', type=int, default=0)

args = parser.parse_args()

hparams_json = f'{args.out_dir}/version_{args.version}/hparam.json'
with open(hparams_json) as f:
    hparams = json.load(f)

import ast
model_type = hparams['model_type'] 
out_channels = ast.literal_eval(hparams['out_channels'])
latent_channels = int(hparams['latent_channels'])
pooling_factor = int(hparams['pooling_factor'])
in_channels = int(hparams['in_channels'])
K = int(hparams['K'])
particles = int(hparams['particles'])
output_particles = int(hparams['output_particles'])
decoder_output = hparams['decoder_output']
mvn_rank = int(hparams['mvn_rank'])
n_blocks = int(hparams['n_blocks'])
substructure = hparams['substructure']
# TODO: Remove this hack
if substructure == 'BrStem':
    filepath = '/vol/biomedic3/bglocker/brainshapes/1000596/T1_first-BrStem_first.vtk'
elif substructure == 'R_Hipp':
    filepath = '/vol/biomedic3/bglocker/brainshapes/1000596/T1_first-R_Hipp_first.vtk'

shape = int(hparams['shape'])
csv_path = hparams['csv_path']

# device
device = 'cuda'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# deterministic
seed = int(hparams['seed'])
np.random.seed(seed)
torch.manual_seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True

# Preprocessor
preprocessor = transforms.get_transforms()

# Load Dataset
mesh_path = '/vol/biomedic3/bglocker/brainshapes'
# cache_path = '/vol/bitbucket/rrr2417/deepscm/deepscm/subexperiments/medical_meshes/notebooks'
cache_path = '.'  # /vol/bitbucket/rrr2417/deepscm/deepscm/submodules/coma_ukbiobank_mesh/'
split = float(hparams['train_test_split'])
substructures = [substructure]
feature_name_map = {
    '31-0.0': 'Sex',
    '21003-0.0': 'Age',
    '25025-2.0': 'Brain Stem Volume',
}

metadata_df = pd.read_csv(csv_path)

total_train_dataset = UKBBMeshDataset(
    mesh_path,
    substructures=substructures,
    split=split,
    train=True,
    transform=preprocessor,
    reload_path=True,
    features_df=metadata_df,
    feature_name_map=feature_name_map,
    cache_path=cache_path,
)
test_dataset = UKBBMeshDataset(
    mesh_path,
    substructures=substructures,
    split=split,
    train=False,
    transform=preprocessor,
    reload_path=True,
    features_df=metadata_df,
    feature_name_map=feature_name_map,
    cache_path=cache_path,
)

val_split = float(hparams['val_split'])
total_train_length = len(total_train_dataset)
val_length = int(val_split * total_train_length)
train_length = total_train_length - val_length

train_dataset, val_dataset = torch.utils.data.random_split(
    total_train_dataset,
    lengths=[train_length, val_length],
    generator=torch.Generator().manual_seed(seed),
)

batch_size = int(hparams['batch_size'])
print(batch_size)
print(total_train_length)
print(len(test_dataset))
train_dataloader = VerticesDataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
)
val_dataloader = VerticesDataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
)
test_dataloader = VerticesDataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)


template = get_data_from_polydata(pv.read(filepath))
model = init_coma(
    model_type,
    template,
    device,
    pooling_factor,
    decoder_output,
    in_channels=in_channels,
    out_channels=out_channels,
    latent_channels=latent_channels,
    K=K, n_blocks=n_blocks,
    mvn_rank=mvn_rank,
)
checkpoint = f'{args.out_dir}/version_{args.version}/checkpoint.pt'
model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
model = model.double()
print()
print('Model Loaded')
print()

# Sanity Check
output_particles = int(hparams['output_particles'])
trial_graph = torch.ones((5, shape, in_channels))
res = model.generate(trial_graph.to(device).double(), output_particles)
print(f'Sanity check, output shape: {res.shape}')
assert res.shape == torch.Size([5, shape, in_channels])

scheduler = StepLR({
    'optimizer': Adam,
    'optim_args': {
        'lr': float(hparams['lr']),
    },
    'step_size': int(hparams['scheduler_steps']),
    'gamma': float(hparams['step_gamma']),
})
loss = CustomELBO(num_particles=int(hparams['particles']))
svi = SVI(model.model, model.guide, scheduler, loss=loss)
svi.loss_class = loss

print('Calculating Train metrics')
metrics, _ = train_eval_svi(svi, model, train_dataloader, device, output_particles, train=False)
print(metrics)
print('Calculating Val metrics')
metrics, _ = train_eval_svi(svi, model, val_dataloader, device, output_particles, train=False)
print(metrics)
print('Calculating Test metrics')
metrics, _ = train_eval_svi(svi, model, test_dataloader, device, output_particles, train=False)
print(metrics)
