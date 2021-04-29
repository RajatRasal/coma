from functools import reduce

import pyro
import numpy as np
import torch
import torch.nn as nn
import pyro.distributions as dist
from torch import Tensor
from pyro.distributions.transforms import (
    neural_autoregressive, ComposeTransform, AffineTransform,
    LowerCholeskyAffine,
)

from .components import (
    Lambda, DeepIndepNormal, DeepLowRankMultivariateNormal,
)


class VAE(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int, 
        decoder_output: str = 'normal',
    ):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.decoder_output = decoder_output

        self.encoder = DeepIndepNormal(encoder, self.latent_dim, latent_dim)
        decoder_flatten = nn.Sequential(
            decoder,
            Lambda(lambda x: x.view(x.shape[0], -1)),
        )
        if decoder_output == 'normal':
            # TODO: Remove hard coding of shape dimensions 
            self.decoder = DeepIndepNormal(decoder_flatten, 642 * 3, 642 * 3)
        elif decoder_output == 'mvn':
            self.decoder = DeepLowRankMultivariateNormal(decoder_flatten, 642 * 3, 642 * 3, 10)
        else:
            raise Exception('Unknown decoder output type')

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
        # Note: plate automatically sets batch size dimension to those which
        #   conditionally independent.
        with pyro.plate('data', x.shape[0]):
            # Define prior distribution p(z) = N(0, I)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.latent_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.latent_dim)))
            z = pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1))
            # Compute p(x|z)
            out_dist = self.decoder.predict(z)  # .to_event(1)
            # TODO: Use pyro condition
            pyro.sample('obs', out_dist, obs=x.view(x.shape[0], -1))

    def guide(self, x):
        """
        Define q(z|x)
        """
        pyro.module('encoder', self.encoder)
        with pyro.plate('data', x.shape[0]):
            z_dist = self.encoder.predict(x, 1)
            pyro.sample('latent', z_dist)

    def generate(self, x, device):
        z_dist = self.encoder.predict(x)
        z = z_dist.sample()
        # TODO: Change this to zeros_like and ones_like
        # [x.shape[0], 642 * 3], requires_grad=False).to(device).double(),
        # torch.ones([x.shape[0], 642 * 3], requires_grad=False).to(device).double(),
        x_base_dist = dist.Normal(
            torch.zeros_like(x, requires_grad=False).view(x.shape[0], -1),
            torch.ones_like(x, requires_grad=False).view(x.shape[0], -1),
        ).to_event(1)
        x_pred_dist = self.decoder.predict(z)

        if self.decoder_output == 'normal':
            transform = AffineTransform(x_pred_dist.mean, x_pred_dist.stddev, 1)
        elif self.decoder_output == 'mvn':
            transform = LowerCholeskyAffine(x_pred_dist.loc, x_pred_dist.scale_tril)
        else:
            raise Exception('Unknown decoder output')
                    
        x_dist = dist.TransformedDistribution(x_base_dist, ComposeTransform([transform]))
        recon = pyro.sample('x', x_dist).view(x.shape[0], 642, 3)
        return recon


class VAE_IAF(VAE):

    def __init__(self, encoder, decoder, latent_dim):
        super().__init__(encoder, decoder, latent_dim)
        hidden_dims = [3 * latent_dim + 1] * 5
        self.iaf1 = neural_autoregressive(latent_dim, hidden_dims=hidden_dims)
        # self.iaf2 = neural_autoregressive(latent_dim, hidden_dims=hidden_dims)
        # self.iaf3 = neural_autoregressive(latent_dim, hidden_dims=hidden_dims)
        self.iaf = [self.iaf1]  # , self.iaf2, self.iaf3]

    def guide(self, x):
        """
        Define q(z|x)
        """
        pyro.module('encoder', self.encoder)
        pyro.module('iaf', nn.ModuleList(self.iaf))

        with pyro.plate('data', x.shape[0]):
            mean, std = self.encoder(x)
            # print(f'mean: {mean.shape}, std: {std.shape}') 
            transformed_z_dist = self.transformed_latent_dist(mean=mean, std=std)
            # print(transformed_z_dist)
            pyro.sample('latent', transformed_z_dist)

    def transformed_latent_dist(self, **kwargs):
        latent_base_dist = dist.Normal(kwargs['mean'], kwargs['std']).to_event(1)
        transformed_z_dist = dist.TransformedDistribution(latent_base_dist, self.iaf)
        return transformed_z_dist
