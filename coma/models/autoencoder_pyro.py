from functools import reduce

import pyro
import torch
import torch.nn as nn
import pyro.distributions as dist
from torch import Tensor
from pyro.distributions.transforms import neural_autoregressive

from .components import Lambda, DeepIndepNormal 


class VAE(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int):
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
        self.decoder = nn.Sequential(
            self.decoder_unit,
            Lambda(lambda x: x.view(x.shape[0], -1)),
            DeepIndepNormal(642 * 3, 642 * 3),
        )
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
        pyro.module('decoder',self.decoder)

        with pyro.plate('data', z.shape[0]):
            # Define prior distribution p(z) = N(0, I)
            z_loc = z.new_zeros(torch.Size((x.shape[0], self.latent_dim)))
            z_scale = z.new_ones(torch.Size((x.shape[0], self.latent_dim)))
            z = pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1))
            # Compute p(x|z)
            recon_dist = self.decoder(z).to_event(1)
            pyro.sample('obs', recon_dist, obs=x.view(x.shape[0], -1))

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

    def guide(self, x):
        """
        Define q(z|x)
        """
        pyro.module('encoder', self.encoder)
        pyro.module('iaf', self.iaf.module)

        with pyro.plate('data', x.shape[0]):
            mean, std = self.encoder(x)
            transformed_z_dist = self.transformed_latent_dist(mean=mean, std=std)
            pyro.sample('latent', transformed_z_dist)

    def transformed_latent_dist(self, **kwargs):
        latent_base_dist = dist.Normal(kwargs['mean'], kwargs['std'])
        transformed_z_dist = dist.TransformedDistribution(
            latent_base_dist,
            [self.iaf]
        ).to_event(1)
        return transformed_z_dist
