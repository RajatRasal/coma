import torch
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import pandas as pd
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from coma.models import init_coma
from coma.datasets.ukbb_meshdata import (
    UKBBMeshDataset, VerticesDataLoader, get_data_from_polydata
)
from coma.utils import transforms
from coma.utils.train_eval_svi import run_svi


parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--out_dir', type=str, default='experiments')

# network hyperparameters
parser.add_argument('--out_channels', nargs='+', default=[32, 32, 32, 64], type=int)
parser.add_argument('--latent_channels', type=int, default=8)
parser.add_argument('--pooling_factor', type=int, default=4)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--particles', type=int, default=3)
parser.add_argument('--decoder_output', type=str, default='normal')

# optimizer hyperparmeters
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--lr_decay', type=float, default=1.0)

# training hyperparameters
parser.add_argument('--train_test_split', type=float, default=0.8)
parser.add_argument('--val_split', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--epochs', type=int, default=300)

# others
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# deterministic
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True

# Preprocessor
preprocessor = transforms.get_transforms()

# Load Dataset
mesh_path = '/vol/biomedic3/bglocker/brainshapes'
cache_path = '/vol/bitbucket/rrr2417/deepscm/deepscm/experiments/medical_meshes/notebooks'
split = args.train_test_split
substructures = ['BrStem']
feature_name_map = {
    '31-0.0': 'Sex',
    '21003-0.0': 'Age',
    '25025-2.0': 'Brain Stem Volume',
}

csv_path = '/vol/biomedic3/bglocker/brainshapes/ukb21079_extracted.csv'
metadata_df = pd.read_csv(csv_path)

total_train_dataset = UKBBMeshDataset(
    mesh_path,
    substructures=substructures,
    split=split,
    train=True,
    transform=preprocessor,
    reload_path=False,
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
    reload_path=False,
    features_df=metadata_df,
    feature_name_map=feature_name_map,
    cache_path=cache_path,
)

val_split = args.val_split
total_train_length = len(total_train_dataset)
val_length = int(val_split * total_train_length)
train_length = total_train_length - val_length

train_dataset, val_dataset = torch.utils.data.random_split(
    total_train_dataset,
    lengths=[train_length, val_length],
    generator=torch.Generator().manual_seed(seed),
)

batch_size = args.batch_size
train_dataloader = VerticesDataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
)
val_dataloader = VerticesDataLoader(
    val_dataset,
    batch_size=10,
    shuffle=False,
)
test_dataloader = VerticesDataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

train_plotting_point = train_dataset.dataset.get_raw(train_dataset.indices[0])
train_data = get_data_from_polydata(train_plotting_point)
template = train_data

in_channels = 3
out_channels = args.out_channels 
latent_channels = args.latent_channels
K = args.K
n_blocks = 1
pooling_factor = args.pooling_factor
decoder_output = args.decoder_output

model = init_coma(
    'vae_svi',
    template,
    device,
    pooling_factor,
    decoder_output,
    in_channels=in_channels,
    out_channels=out_channels,
    latent_channels=latent_channels,
    K=K, n_blocks=n_blocks,
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
trial_graph = torch.ones((5, 642, 3))
res = model.generate(trial_graph.to(device).double(), device)
print(f'Sanity check, output shape: {res.shape}')

optimiser = Adam({'lr': args.lr})
loss = Trace_ELBO(num_particles=args.particles)
svi = SVI(model.model, model.guide, optimiser, loss=loss)

# TODO: Save hyperparameters
# Save model weights

"""
GCN encoder is severely underparametrised. Linear is needed.

VAE - 50 epochs, lr = 1e-5, batch_size = 10
VAE_IAF with 3 IAFs lr = 1e-5 batch = 50 particles = 3
    Too many IAF units becomes unstable and diverges > 3
VAE - 50 epochs, lr = 1e-3, batch_size = 50 particles = 3

Next up:
VAE - 50 epochs, lr = 1e-3, batch_size = 50, with MVN decoder
"""
print(f'Total epochs: {args.epochs}')
for i in range(args.epochs):
    print('Epoch no:', i)
    run_svi(svi, train_dataloader, val_dataloader, 1, device)
    for batch in val_dataloader:
        pred = model.generate(batch.x.to(device)[0].view(1, 642, 3), device)
        print(batch.x[0])
        print(pred)
        print()
        break
