import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch_geometric.nn import ChebConv, GCNConv
from torch_geometric.data import Data
from torch_scatter import scatter_add

from coma.utils import mesh_sampling, utils
from .inits import reset


def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class Enblock(nn.Module):
    def __init__(self, in_channels, out_channels, K, n_blocks=1, **kwargs):
        super(Enblock, self).__init__()
        assert n_blocks > 0
        self.blocks = torch.nn.ModuleList()
        for i in range(n_blocks):
            _in_channels = in_channels if i == 0 else out_channels
            self.blocks.append(
                ChebConv(_in_channels, out_channels, K, **kwargs)
            )
        # self.conv = ChebConv(in_channels, out_channels, K, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.blocks.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x, edge_index, down_transform):
        for layer in self.blocks:
            x = layer(x, edge_index)
            # print(x.shape)
        out = F.elu(x, inplace=True)
        out = Pool(out, down_transform)
        return out


class Deblock(nn.Module):
    def __init__(self, in_channels, out_channels, K, n_blocks=1, **kwargs):
        super(Deblock, self).__init__()
        assert n_blocks > 0
        self.blocks = torch.nn.ModuleList()
        for i in range(n_blocks):
            _in_channels = in_channels if i == 0 else out_channels
            self.blocks.append(
                ChebConv(_in_channels, out_channels, K, **kwargs)
            )
        # self.conv = ChebConv(in_channels, out_channels, K, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.blocks.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x, edge_index, up_transform):
        out = Pool(x, up_transform)
        for layer in self.blocks:
            out = layer(out, edge_index)
        out = F.elu(out, inplace=True)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, edge_index,
        down_transform, up_transform, K, n_blocks, **kwargs
    ):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_index = edge_index
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_verts = self.down_transform[-1].size(0)

        self.layers = nn.ModuleList()

        for idx in range(len(out_channels)):
            in_channels = in_channels if not idx else out_channels[idx - 1]
            block = Enblock(in_channels, out_channels[idx], K, n_blocks, **kwargs)
            self.layers.append(block)

        self.layers.append(
            nn.Linear(self.num_verts * out_channels[-1], latent_channels)
        )

        self.reset_parameters()

    def get_output_shape(self):
        return 

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i != len(self.layers) - 1:
                x = layer(x, self.edge_index[i], self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, edge_index,
        down_transform, up_transform, K, n_blocks, **kwargs
    ):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_index = edge_index
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_verts = self.down_transform[-1].size(0)

        self.layers = nn.ModuleList()

        self.layers.append(
            nn.Linear(latent_channels, self.num_verts * out_channels[-1])
        )
        for idx in range(len(out_channels)):
            in_channels = out_channels[-idx - 1] if not idx else out_channels[-idx]
            block = Deblock(in_channels, out_channels[-idx - 1], K, n_blocks, **kwargs)
            self.layers.append(block)

        # reconstruction
        self.layers.append(
            ChebConv(self.out_channels[0], self.in_channels, K, **kwargs)
        )

        self.reset_parameters()

    def get_output_shape(self):
        pass

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        num_layers = len(self.layers)
        num_deblocks = num_layers - 2
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_verts, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.edge_index[num_deblocks - i],
                    self.up_transform[num_deblocks - i])
            else:
                # last layer
                x = layer(x, self.edge_index[0])
        return x


class IndepNormal(nn.Module):

    def __init__(self, backbone: nn.Module):
        super(IndepNormal, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        return x, torch.ones_like(x)
    
    def predict(self, x, event_ndim=None) -> dist.Normal:
        mean, std = self(x)
        if event_ndim is None:
            event_ndim = len(mean.shape[1:])  # keep only batch dimension
        return dist.Normal(mean, std).to_event(event_ndim)


class IndepNormalDeepMean(nn.Module):

    def __init__(self, backbone: nn.Module, hidden_dim: int, out_dim: int):
        super(IndepNormalDeepMean, self).__init__()
        self.backbone = backbone
        self.mean_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        mean = self.mean_head(x)
        # TODO: Remove harcoding of std
        std = torch.ones_like(mean) * 1e-9 
        return mean, std
    
    def predict(self, x, event_ndim=None) -> dist.Normal:
        mean, std = self(x)
        if event_ndim is None:
            event_ndim = len(mean.shape[1:])  # keep only batch dimension
        return dist.Normal(mean, std).to_event(event_ndim)


class IndepNormalDeepVar(nn.Module):

    def __init__(self, backbone: nn.Module, hidden_dim: int, out_dim: int):
        super(IndepNormalDeepVar, self).__init__()
        self.backbone = backbone
        self.logvar_head = nn.Linear(hidden_dim, out_dim)

    def __logvar_to_std(self, logvar):
        return (0.5 * logvar).exp()
    
    def forward(self, x):
        x = self.backbone(x)
        logvar = self.logvar_head(x)
        std = self.__logvar_to_std(logvar)
        return x, std
    
    def predict(self, x, event_ndim=None) -> dist.Normal:
        mean, std = self(x)
        if event_ndim is None:
            event_ndim = len(mean.shape[1:])  # keep only batch dimension
        return dist.Normal(mean, std).to_event(event_ndim)


class DeepIndepNormal(nn.Module):
    """
    Code taken from DeepSCM repo
    """
    def __init__(self, backbone: nn.Module, hidden_dim: int, out_dim: int):
        super(DeepIndepNormal, self).__init__()
        self.backbone = backbone
        self.mean_head = nn.Linear(hidden_dim, out_dim)
        self.logvar_head = nn.Linear(hidden_dim, out_dim)

    def __logvar_to_std(self, logvar):
        return (0.5 * logvar).exp()
    
    def forward(self, x):
        x = self.backbone(x)
        # print(x.shape)
        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        std = self.__logvar_to_std(logvar)
        # print(mean.shape, std.shape)
        return mean, std 
    
    def predict(self, x, event_ndim=None) -> dist.Normal:
        mean, std = self(x)
        if event_ndim is None:
            event_ndim = len(mean.shape[1:])  # keep only batch dimension
        return dist.Normal(mean, std).to_event(event_ndim)


class DeepConv1dIndepNormal(nn.Module):
    """
    Code taken from DeepSCM repo
    """
    def __init__(self, backbone: nn.Module, hidden_dim: int, out_dim: int,
        filters: int, kernel_size: int, padding: int,
    ):
        super(DeepConv1dIndepNormal, self).__init__()
        self.backbone = backbone
        self.mean_head = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size, padding=padding),
            nn.Conv1d(filters, 1, 1),
        )
        self.logvar_head = nn.Sequential( 
            nn.Conv1d(1, filters, kernel_size, padding=padding),
            nn.Conv1d(filters, 1, 1),
        )

    def __logvar_to_std(self, logvar):
        return (0.5 * logvar).exp()
    
    def forward(self, x):
        x = self.backbone(x)
        shape = x.shape
        x = x.reshape(shape[0], 1, shape[1])
        mean = self.mean_head(x).reshape(shape[0], shape[1])
        logvar = self.logvar_head(x).reshape(shape[0], shape[1])
        std = self.__logvar_to_std(logvar)
        return mean, std
    
    def predict(self, x, event_ndim=None) -> dist.Normal:
        mean, std = self(x)
        if event_ndim is None:
            event_ndim = len(mean.shape[1:])  # keep only batch dimension
        return dist.Normal(mean, std).to_event(event_ndim)


class DeepLowRankMultivariateNormal(nn.Module):
    """
    Code taken from DeepSCM repo
    """
    def __init__(self, backbone: nn.Module, hidden_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.backbone = backbone
        self.out_dim = out_dim
        self.rank = rank
        self.mean_head = nn.Linear(hidden_dim, out_dim)
        self.factor_head = nn.Linear(hidden_dim, out_dim * rank)
        self.logdiag_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.backbone(x)
        mean = self.mean_head(h)
        diag = self.logdiag_head(h).exp()
        # (0.5 * logvar).exp()
        factors = self.factor_head(h).view(x.shape[0], self.out_dim, self.rank)
        return mean, diag, factors

    def predict(self, x) -> dist.LowRankMultivariateNormal:
        mean, diag, factors = self(x)
        return dist.LowRankMultivariateNormal(mean, factors, diag)


class DeepMultivariateNormal(nn.Module):
    """
    Code taken from DeepSCM repo
    """
    def __init__(self, backbone: nn.Module, hidden_dim: int, out_dim: int):
        super().__init__()
        self.backbone = backbone
        cov_lower_dim = (out_dim * (out_dim - 1)) // 2
        self.mean_head = nn.Linear(hidden_dim, out_dim)
        self.lower_head = nn.Linear(hidden_dim, cov_lower_dim)
        self.logdiag_head = nn.Linear(hidden_dim, out_dim)

    def __assemble_tril(self, diag: torch.Tensor, lower_vec: torch.Tensor) -> torch.Tensor:
        dim = diag.shape[-1]
        L = torch.diag_embed(diag)  # L is lower-triangular
        i, j = torch.tril_indices(dim, dim, offset=-1)
        L[..., i, j] = lower_vec
        return L

    def forward(self, x):
        h = self.backbone(x)
        mean = self.mean_head(h)
        diag = self.logdiag_head(h).exp()
        lower = self.lower_head(h)
        scale_tril = self.__assemble_tril(diag, lower)
        return mean, scale_tril 

    def predict(self, x) -> dist.MultivariateNormal:
        mean, scale_tril = self(x)
        return dist.MultivariateNormal(mean, scale_tril=scale_tril) 


class GCNDeepIndepNormal(nn.Module):
    """
    Code taken from DeepSCM repo
    """
    def __init__(self, backbone: nn.Module, hidden_channels: int, out_channels: int, edge_index: Tensor):
        super(GCNDeepIndepNormal, self).__init__()
        self.backbone = backbone
        self.mean_head = GCNConv(hidden_channels, out_channels)
        self.logvar_head = GCNConv(hidden_channels, out_channels)
        self.edge_index = edge_index

    def __logvar_to_std(self, logvar):
        return (0.5 * logvar).exp()

    def forward(self, x):
        x = self.backbone(x)
        mean = self.mean_head(x, self.edge_index)
        logvar = self.logvar_head(x, self.edge_index)
        std = self.__logvar_to_std(logvar)
        return mean, std 
    
    def predict(self, x, event_ndim=None) -> dist.Normal:
        mean, std = self(x)
        if event_ndim is None:
            event_ndim = len(mean.shape[1:])  # keep only batch dimension
        return dist.Normal(mean, std).to_event(event_ndim)


class Lambda(torch.nn.Module):
    """
    Code taken from DeepSCM repo
    """
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x)
