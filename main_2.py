import torch
import pandas as pd
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from coma.models import init_coma
from coma.datasets.ukbb_meshdata import (
	UKBBMeshDataset, VerticesDataLoader, get_data_from_polydata
)
from coma.utils import transforms
from coma.utils.train_eval_svi import run_svi


preprocessor = transforms.get_transforms()

mesh_path = '/vol/biomedic3/bglocker/brainshapes'
cache_path = '/vol/bitbucket/rrr2417/deepscm/deepscm/experiments/medical_meshes/notebooks'
split = 0.8
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
    cache_path = cache_path,
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
    cache_path = cache_path,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_split = 0.1
total_train_length = len(total_train_dataset)
val_length = int(val_split * total_train_length)
train_length = total_train_length - val_length

train_dataset, val_dataset = torch.utils.data.random_split(
    total_train_dataset,
    lengths=[train_length, val_length],
    generator=torch.Generator().manual_seed(42),
)

batch_size = 10
train_dataloader = VerticesDataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
)
val_dataloader = VerticesDataLoader(
    val_dataset,
    batch_size=20,
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
out_channels = [32, 32, 32, 64]
latent_channels = 20
K = 10
n_blocks = 1
pooling_factor = 4

model = init_coma(
    'vae_iaf_svi',
    template,
    device,
    pooling_factor,
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

# trial_graph = torch.ones((5, 642, 3))
# res = model.generate(trial_graph.to(device).double())

optimiser = Adam({'lr': 1e-3})
loss = Trace_ELBO(num_particles=3)
svi = SVI(model.model, model.guide, optimiser, loss=loss)

run_svi(svi, train_dataloader, val_dataloader, 10, device)
