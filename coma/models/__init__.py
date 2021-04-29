import psbody.mesh
from torch_geometric.data import Data

from .autoencoder_pyro import VAE as VAE_SVI, VAE_IAF as VAE_IAF_SVI
from .autoencoder import AE, VAE
from .components import Encoder, Decoder
from coma.utils import mesh_sampling, utils


def init_coma(model_type: str, template: Data, device: str, pooling_factor: int = 4,
    decoder_output: str = 'normal', **kwargs
):
    mesh = psbody.mesh.Mesh(
        v=template.pos.detach().cpu().numpy(),
        f=template.face.T.detach().cpu().numpy(),
    )
    ds_factors = [pooling_factor] * 4  # 4, 4, 4, 4]
    _, A, D, U, F = mesh_sampling.generate_transform_matrices(mesh, ds_factors)
    tmp = {'face': F, 'adj': A, 'down_transform': D, 'up_transform': U}

    edge_index_list = [
        utils.to_edge_index(adj).to(device)
        for adj in tmp['adj']
    ]
    down_transform_list = [
        utils.to_sparse(down_transform).to(device)
        for down_transform in tmp['down_transform']
    ]
    up_transform_list = [
        utils.to_sparse(up_transform).to(device)
        for up_transform in tmp['up_transform']
    ]

    encoder = Encoder(
        **kwargs,
        edge_index=edge_index_list,
        down_transform=down_transform_list,
        up_transform=up_transform_list,
    )
    decoder = Decoder(
        **kwargs,
        edge_index=edge_index_list,
        down_transform=down_transform_list,
        up_transform=up_transform_list,
    )
    latent_dim = kwargs['latent_channels']

    models = {
        'ae': AE,
        'vae': VAE,
        'vae_svi': VAE_SVI,
        'vae_iaf_svi': VAE_IAF_SVI,
    }
    model = models[model_type](encoder, decoder, latent_dim, decoder_output)
    model = model.to(device)
    return model


__all__ = [AE, VAE, VAE_SVI, VAE_IAF_SVI, init_coma]
