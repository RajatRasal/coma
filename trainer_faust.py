import torch
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import pandas as pd
import pyvista as pv
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import StepLR
from torch.optim import Adam
from torch_geometric.utils import to_trimesh
from torch_geometric.datasets import FAUST
from torch_geometric.data import DataLoader 

from coma.models import init_coma
from coma.models.elbo import CustomELBO
from coma.datasets.ukbb_meshdata import (
    UKBBMeshDataset, VerticesDataLoader, get_data_from_polydata
)
from coma.datasets.faust import FAUSTDataLoader, FullFAUST, split_faust_by_person
from coma.utils import writer
from coma.utils.train_eval_svi import run_svi


parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--out_dir', type=str, default='experiments')
parser.add_argument('--exp_name', type=str, default='faust')

# network hyperparameters
parser.add_argument('--model_type', default='vae_svi', type=str)
parser.add_argument('--out_channels', nargs='+', default=[16, 16, 16, 32], type=int)
parser.add_argument('--latent_channels', type=int, default=8)
parser.add_argument('--pooling_factor', type=int, default=4)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--particles', type=int, default=1)
parser.add_argument('--output_particles', type=int, default=10)
parser.add_argument(
    '--decoder_output',
    default='normal',
    choices=['normal', '_normal', 'low_rank_mvn', 'mvn', 'conv_normal', 'deepvar', 'deepmean'],
)
parser.add_argument('--mvn_rank', type=int, default=10)
parser.add_argument('--filters', type=int, default=10)
parser.add_argument('--n_blocks', type=int, default=1)

# optimizer hyperparmeters
parser.add_argument('--lr', type=float, default=1e-3)

# training hyperparameters
parser.add_argument('--train_test_split', type=float, default=0.8)
parser.add_argument('--val_split', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--scheduler_steps', type=int, default=50)
parser.add_argument('--step_gamma', type=float, default=0.1)

# data arguments
parser.add_argument(
    '--datasets',
    default='faust',
    choices=['faust', 'full_faust', 'both']
)

# 
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu_no', type=int, default=0, choices=[0, 1])

args = parser.parse_args()

# device
device = torch.device(f'cuda:{args.gpu_no}' if torch.cuda.is_available() else 'cpu')
print('DEVICE')
print(device)

# deterministic
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True

batch_size = args.batch_size

# Preprocessor
# preprocessor = transforms.get_transforms()
val_split = args.val_split

if args.datasets == 'faust':
    train_dataset = FAUST('.')
    val_dataset = FAUST('.', train=False)
    train_dataloader = FAUSTDataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = FAUSTDataLoader(val_dataset, batch_size=batch_size)
elif args.datasets == 'full_faust':
    dataset = FullFAUST('.')
    # TODO: Remove hardcoding of test person
    train_dataset, val_dataset = split_faust_by_person(dataset, [1]) 
    train_dataloader = FAUSTDataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = FAUSTDataLoader(val_dataset, batch_size=batch_size)
else:
    raise NotImplementedError('')

# for DFAUST splitting. 9:1 ratio, with seed from above

print('Template')
template = train_dataset[0]
pv_template = pv.wrap(to_trimesh(template))
print(template)
print()

print('Initialise writer')
writer = writer.MeshWriter(args, pv_template)

shape = 6890
in_channels = 3
# For 1D conv decoder
kernel_size = 5031 
# max(
#     template.face.T.max(axis=1).values - template.face.T.min(axis=1).values
# )
padding = kernel_size // 2


model = init_coma(
    args.model_type,
    template,
    device,
    shape,
    args.pooling_factor,
    args.decoder_output,
    in_channels=in_channels,
    out_channels=args.out_channels,
    latent_channels=args.latent_channels,
    K=args.K,
    n_blocks=args.n_blocks,
    mvn_rank=args.mvn_rank,
    filters=args.filters,
    kernel_size=kernel_size,
    padding=padding,
)
model = model.double()
print()
print(model)
print()

total_params = sum(p.numel() for p in model.parameters())
print()
print(total_params)
print()

# Sanity Check
output_particles = args.output_particles
trial_graph = torch.ones((5, shape, in_channels))
res = model.generate(trial_graph.to(device).double(), output_particles)
print(f'Sanity check, output shape: {res.shape}')
assert res.shape == torch.Size([5, shape, in_channels])

optimiser = Adam
scheduler = StepLR({
    'optimizer': optimiser,
    'optim_args': {
        'lr': args.lr,
    },
    'step_size': args.scheduler_steps,
    'gamma': args.step_gamma,
    'verbose': True,
})
loss = CustomELBO(num_particles=args.particles)
svi = SVI(model.model, model.guide, scheduler, loss=loss)
svi.loss_class = loss

run_svi(
    svi, model, train_dataloader, val_dataloader,
    args.epochs, scheduler, device, output_particles, writer
)
