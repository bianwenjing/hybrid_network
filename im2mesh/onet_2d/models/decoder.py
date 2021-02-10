import torch.nn as nn
import torch.nn.functional as F
from im2mesh.layers import (
    ResnetBlockFC, CResnetBlockConv1d,
    CBatchNorm1d, CBatchNorm1d_legacy,
    ResnetBlockConv1d
)
from im2mesh.common import map2local
from im2mesh.pos_encoding import encode_position
import torch

class Decoder(nn.Module):
    ''' Decoder class.

    It does not perform any form of normalization.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, z_resolution, dim=2, z_dim=128, c_dim=128,
                 hidden_size=128, leaky=False, attention=False, positional_encoding=False, camera=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        self.fc_p = nn.Linear(dim, hidden_size)

        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        if not c_dim == 0:
            self.fc_c = nn.Linear(c_dim, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)

        self.fc_out = nn.Linear(hidden_size, z_resolution)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c=None, **kwargs):
        batch_size, T, D = p.size()

        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(1)
            net = net + net_z

        if self.c_dim != 0:
            # print('((((((', c.shape)  # [64, 256]
            net_c = self.fc_c(c).unsqueeze(1)
            # print('@@@@@@@@@@@@', net.shape, net_c.shape)  # [64, 127, 128] # [64, 1, 128]
            net = net + net_c
            # print('##', net.shape) # [64, 127, 128]
        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(net))
        # out = out.squeeze(-1)

        return out


class DecoderCBatchNorm(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(self, z_resolution, dim=2, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False, legacy=False, attention=False, positional_encoding=False, camera=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        ########positional encoding###########
        self.postional_encoding = positional_encoding
        self.camera = camera

        if positional_encoding:
            # unit_size = 0.1
            # self.map2local = map2local(unit_size, pos_encoding='sin_cos')
            self.fc_p = nn.Linear(42, hidden_size)
        else:
            self.fc_p = nn.Linear(dim, hidden_size)
            # self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        if camera:
            c_dim = c_dim + hidden_size
            self.fc_cam = nn.Linear(63, hidden_size)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, z_resolution, 1)
        # self.fc_out = nn.Linear(hidden_size, z_resolution)

        ###########attention
        self.attention = attention
        if attention:
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        ########################


        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        if self.postional_encoding:
            p = p.float()
            # p = self.map2local(p)
            # from [-0.55, 0.55] to [-1, 1]
            p = p*2/1.1
            p = encode_position(p, 10)

        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        if self.camera:
            c, camera_position = c[:, :self.c_dim], c[:, self.c_dim:]
            camera_position = F.normalize(camera_position, dim = 1)
            camera_position = encode_position(camera_position, 10)
            camera_position = self.fc_cam(camera_position)
            c = F.normalize(c, dim=1)
            c = torch.cat([c, camera_position], dim=1)

        net = net.transpose(1, 2)
        # batch_size, D, T = p.size()
        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        net = net.transpose(1,2)
        # attention
        if self.attention:
            net = self.transformer_encoder(net)

        net = net.transpose(1,2)
        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.transpose(1,2)

        return out

class DecoderCBatchNorm_pe(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(self, z_resolution, dim=2, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False, legacy=False, attention=False, positional_encoding=False):
        super().__init__()
        self.z_dim = z_dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        # self.fc_p = nn.Linear(60, hidden_size)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)


        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, z_resolution, 1)
        # self.fc_out = nn.Linear(hidden_size, z_resolution)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()

        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)


        # print('#############', net.shape, self.actvn(self.bn(net, c)).shape) (64, 256, 1024)
        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.transpose(1,2)

        return out

class DecoderCBatchNorm2(nn.Module):
    ''' Decoder with CBN class 2.

    It differs from the previous one in that the number of blocks can be
    chosen.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of ResNet blocks
    '''

    def __init__(self, dim=3, z_dim=0, c_dim=128,
                 hidden_size=256, n_blocks=5):
        super().__init__()
        self.z_dim = z_dim
        if z_dim != 0:
            self.fc_z = nn.Linear(z_dim, c_dim)

        self.conv_p = nn.Conv1d(dim, hidden_size, 1)
        self.blocks = nn.ModuleList([
            CResnetBlockConv1d(c_dim, hidden_size) for i in range(n_blocks)
        ])

        self.bn = CBatchNorm1d(c_dim, hidden_size)
        self.conv_out = nn.Conv1d(hidden_size, 1, 1)
        self.actvn = nn.ReLU()

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.conv_p(p)

        if self.z_dim != 0:
            c = c + self.fc_z(z)

        for block in self.blocks:
            net = block(net, c)

        out = self.conv_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out


class DecoderCBatchNormNoResnet(nn.Module):
    ''' Decoder CBN with no ResNet blocks class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.fc_0 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.fc_1 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.fc_2 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.fc_3 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.fc_4 = nn.Conv1d(hidden_size, hidden_size, 1)

        self.bn_0 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_1 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_2 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_3 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_4 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_5 = CBatchNorm1d(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net = self.actvn(self.bn_0(net, c))
        net = self.fc_0(net)
        net = self.actvn(self.bn_1(net, c))
        net = self.fc_1(net)
        net = self.actvn(self.bn_2(net, c))
        net = self.fc_2(net)
        net = self.actvn(self.bn_3(net, c))
        net = self.fc_3(net)
        net = self.actvn(self.bn_4(net, c))
        net = self.fc_4(net)
        net = self.actvn(self.bn_5(net, c))
        out = self.fc_out(net)
        out = out.squeeze(1)

        return out


class DecoderBatchNorm(nn.Module):
    ''' Decoder with batch normalization class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        if self.c_dim != 0:
            self.fc_c = nn.Linear(c_dim, hidden_size)
        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = ResnetBlockConv1d(hidden_size)
        self.block1 = ResnetBlockConv1d(hidden_size)
        self.block2 = ResnetBlockConv1d(hidden_size)
        self.block3 = ResnetBlockConv1d(hidden_size)
        self.block4 = ResnetBlockConv1d(hidden_size)

        self.bn = nn.BatchNorm1d(hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        if self.c_dim != 0:
            net_c = self.fc_c(c).unsqueeze(2)
            net = net + net_c

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(self.bn(net)))
        out = out.squeeze(1)

        return out
