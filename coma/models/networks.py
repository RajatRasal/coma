import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_scatter import scatter_add

from .inits import reset


def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class Enblock(nn.Module):
    def __init__(self, in_channels, out_channels, K, **kwargs):
        super(Enblock, self).__init__()
        self.conv = ChebConv(in_channels, out_channels, K, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.conv.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x, edge_index, down_transform):
        out = F.elu(self.conv(x, edge_index))
        out = Pool(out, down_transform)
        return out


class Deblock(nn.Module):
    def __init__(self, in_channels, out_channels, K, **kwargs):
        super(Deblock, self).__init__()
        self.conv = ChebConv(in_channels, out_channels, K, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.conv.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x, edge_index, up_transform):
        out = Pool(x, up_transform)
        out = F.elu(self.conv(out, edge_index))
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, edge_index,
        down_transform, up_transform, K, num_verts, **kwargs
    ):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_index = edge_index
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_verts = num_verts

        self.layers = nn.ModuleList()

        for idx in range(len(out_channels)):
            in_channels = in_channels if not idx else out_channels[idx - 1]
            block = Enblock(in_channels, out_channels[idx], K, **kwargs)
            self.layers.append(block)

        self.layers.append(
            nn.Linear(self.num_verts * out_channels[-1], latent_channels)
        )

        self.reset_parameters()

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
        down_transform, up_transform, K, num_verts, **kwargs
    ):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_index = edge_index
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_verts = num_verts

        self.layers = nn.ModuleList()

        self.layers.append(
            nn.Linear(latent_channels, self.num_verts * out_channels[-1])
        )
        for idx in range(len(out_channels)):
            in_channels = out_channels[-idx - 1] if not idx else out_channels[-idx]
            block = Deblock(in_channels, out_channels[-idx - 1], K, **kwargs)
            self.layers.append(block)

        # reconstruction
        print(self.in_channels)
        self.layers.append(
            ChebConv(self.out_channels[0], self.in_channels, K, **kwargs)
        )

        self.reset_parameters()

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


class AE(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, edge_index,
                 down_transform, up_transform, K, Encoder, Decoder, **kwargs):
        super(AE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_index = edge_index
        self.down_transform = down_transform
        self.up_transform = up_transform
        # self.num_vert used in the last and the first layer of encoder and decoder
        self.num_verts = self.down_transform[-1].size(0)

        self.encoder = Encoder(in_channels, out_channels, latent_channels,
            edge_index, down_transform, up_transform, K, self.num_verts, **kwargs,
        )
        self.decoder = Decoder(in_channels, out_channels, latent_channels,
            edge_index, down_transform, up_transform, K, self.num_verts, **kwargs,
        )

        # encoder
        """
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    Enblock(in_channels, out_channels[idx], K, **kwargs))
            else:
                self.en_layers.append(
                    Enblock(out_channels[idx - 1], out_channels[idx], K,
                            **kwargs))
        self.en_layers.append(
            nn.Linear(self.num_vert * out_channels[-1], latent_channels))
        """

        # decoder
        """
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(latent_channels, self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(
                    Deblock(out_channels[-idx - 1], out_channels[-idx - 1], K,
                            **kwargs))
            else:
                self.de_layers.append(
                    Deblock(out_channels[-idx], out_channels[-idx - 1], K,
                            **kwargs))
        # reconstruction
        self.de_layers.append(
            ChebConv(out_channels[0], in_channels, K, **kwargs))
        """

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    """
    def encoder(self, x):
        for i, layer in enumerate(self.en_layers):
            if i != len(self.en_layers) - 1:
                x = layer(x, self.edge_index[i], self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        return x

    def decoder(self, x):
        num_layers = len(self.de_layers)
        num_deblocks = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.edge_index[num_deblocks - i],
                          self.up_transform[num_deblocks - i])
            else:
                # last layer
                x = layer(x, self.edge_index[0])
        return x
    """

    def forward(self, x):
        # x - batched feature matrix
        z = self.encoder(x)
        out = self.decoder(z)
        return out
