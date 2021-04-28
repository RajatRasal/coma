import torch.nn as nn

from torch import Tensor


class AE(nn.Module):
    """
    def __init__(self, in_channels, out_channels, latent_channels, edge_index,
                 down_transform, up_transform, K, n_blocks, Encoder, Decoder, **kwargs):
        super(AE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_index = edge_index
        self.down_transform = down_transform
        self.up_transform = up_transform
        # self.num_vert used in the last and the first layer of encoder and decoder
        self.num_verts = self.down_transform[-1].size(0)

        self.encoder = Encoder(in_channels, out_channels, latent_channels,
            edge_index, down_transform, up_transform, K, self.num_verts, n_blocks, **kwargs,
        )
        self.decoder = Decoder(in_channels, out_channels, latent_channels,
            edge_index, down_transform, up_transform, K, self.num_verts, n_blocks, **kwargs,
        )
    """

    def __init__(self, encoder, decoder, latent_dim):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reset_parameters()

    """
    @classmethod
    def init_coma(cls, template: Data, device: str, pooling_factor: int = 4, **kwargs):
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

        return cls(
            **kwargs,
            edge_index=edge_index_list,
            down_transform=down_transform_list,
            up_transform=up_transform_list,
        ).to(device)
    """

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        # x - batched feature matrix
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class VAE(nn.Module):

    def __init__(self, encoder, decoder, latent_dim):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loc = nn.Linear(latent_dim, latent_dim)
        self.scale = nn.Linear(latent_dim, latent_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def reparametrise(self, mean: Tensor, log_std: Tensor) -> Tensor:
        # TODO: Change this to using multiple MC particles
        eps = torch.randn_like(std)
        z = mean + eps * log_std.exp()
        return z

    def _gaussian_parameters(self, enc: Tensor) -> [Tensor, Tensor]:
        mean = self.loc(enc)
        log_std = self.scale(enc)
        return mean, log_std
    
    def forward(self, x: Tensor) -> [Tensor, Tensor, Tensor, Tensor]:
        enc = self.encode(x)
        mean, log_std = self._gaussian_parameters(enc)
        z = self.reparametrise(mean, log_std)
        return self.decode(z), z, mean, log_std
    
    def generate(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]
    
    @staticmethod
    def loss_function(preds: Tensor, targets: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
        log_prob = F.mse_loss(preds, targets) 
        
        kl_loss = torch.sum(1 + 2 * log_std - mean ** 2 - log_std.exp() ** 2, dim=1)
        kl_loss = -0.5 * torch.mean(kl_loss)
        
        loss = log_prob + kl_loss
        
        return loss, log_prob, kl_loss
