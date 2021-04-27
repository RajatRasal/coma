from functools import reduce

import pyro
import torch
import torch.nn as nn
import pyro.distributions as dist
from torch import Tensor
from pyro.distributions.transforms import neural_autoregressive

from .components import Lambda, DeepIndepNormal, GCNDeepIndepNormal


class VAE(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int, 
        decoder_output: str = 'GCN',
    ):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.encoder_unit = encoder
        self.decoder_unit = decoder

        # enc_out_shape = encoder.get_output_shape()
        # self.dec_out_shape = decoder.get_output_shape()
        # enc_out_shape_flat = reduce(lambda x, y: x * y, enc_out_shape)
        # flatten_view = BatchPreservingView(self.dec_out_shape, enc_out_shape_flat)
        # unflatten_view = BatchPreservingView(enc_out_shape_flat, enc_out_shape)

        # TODO: Remove hard coding of shape dimensions 
        self.encoder = nn.Sequential(
            self.encoder_unit,
            DeepIndepNormal(self.latent_dim, self.latent_dim),
        )
        if decoder_output == 'linear':
            self.decoder = nn.Sequential(
                self.decoder_unit,
                Lambda(lambda x: x.view(x.shape[0], -1)),
                DeepIndepNormal(642 * 3, 642 * 3),
            )
        elif decoder_output == 'GCN':
            self.decoder = nn.Sequential(
                self.decoder_unit,
                # TODO: Remove the edge_index hack
                GCNDeepIndepNormal(3, 3, self.encoder_unit.edge_index[0]),
                Lambda(lambda params: (
                    params[0].view(params[0].shape[0], -1),
                    params[1].view(params[1].shape[0], -1))),
            )
        else:
            assert 'Unknown decoder output type'
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def model(self, x):
        """
        Defines p(x|z)p(z) - decoder + latent prior
        """
        pyro.module('decoder', self.decoder)

        with pyro.plate('data', x.shape[0]):
            # Define prior distribution p(z) = N(0, I)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.latent_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.latent_dim)))
            # print(z_loc.shape, z_scale.shape)
            z = pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1))
            # Compute p(x|z)
            # print('x', x.shape)
            # print('latent:', z.shape)
            mean, std = self.decoder(z)
            # print(f'mean: {mean.shape}, std: {std.shape}')
            pyro.sample('obs', dist.Normal(mean, std).to_event(1), obs=x.view(x.shape[0], -1))

    def guide(self, x):
        """
        Define q(z|x)
        """
        pyro.module('encoder', self.encoder)

        with pyro.plate('data', x.shape[0]):
            mean, std = self.encoder(x)
            # Event 1 from right because batch size is in the 0th
            # index and every batch is conditionally independent.
            # Latent dimensions are in the first index and are dependent. 
            z_dist = self.transformed_latent_dist(mean=mean, std=std)
            # print(f'z_dist: {z_dist}')
            pyro.sample('latent', z_dist)

    def transformed_latent_dist(self, **kwargs):
        z_dist = dist.Normal(kwargs['mean'], kwargs['std'])
        return z_dist.to_event(1)

    def generate(self, x):
        mean, std = self.encoder(x)
        z_dist = self.transformed_latent_dist(mean=mean, std=std)
        z = z_dist.sample()
        recon = self.decoder_unit(z)
        return recon


class VAE_IAF(VAE):

    def __init__(self, encoder, decoder, latent_dim):
        super().__init__(encoder, decoder, latent_dim)
        hidden_dims = [3 * latent_dim + 1] * 3
        self.iaf = neural_autoregressive(latent_dim, hidden_dims=hidden_dims)
        # self.iaf = [af]

    def guide(self, x):
        """
        Define q(z|x)
        """
        pyro.module('encoder', self.encoder)
        pyro.module('iaf', self.iaf)

        with pyro.plate('data', x.shape[0]):
            mean, std = self.encoder(x)
            # print(f'mean: {mean.shape}, std: {std.shape}') 
            transformed_z_dist = self.transformed_latent_dist(mean=mean, std=std)
            # print(transformed_z_dist)
            pyro.sample('latent', transformed_z_dist)

    def transformed_latent_dist(self, **kwargs):
        latent_base_dist = dist.Normal(kwargs['mean'], kwargs['std']).to_event(1)
        transformed_z_dist = dist.TransformedDistribution(latent_base_dist, [self.iaf])
        return transformed_z_dist
