import torch
import torch.nn as nn
import torch.nn.functional as F
from im2mesh.common import normalize_coordinate # normalize_3d_coordinate
from im2mesh.layers import (
    ResnetBlockFC, CResnetBlockConv1d,
    CBatchNorm1d, CBatchNorm1d_legacy,
    ResnetBlockConv1d
)
from im2mesh.pos_encoding import encode_position

class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=2, c_dim_local=128, c_dim_global=128, z_resolution = 32,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1, normalize=False, positional_encoding=False):
        super().__init__()
        ################################3
        self.fc_global = nn.Linear(256, c_dim_global)
        ####################################
        self.c_dim_local = c_dim_local
        self.c_dim_global = c_dim_global
        # c_dim = c_dim_local + c_dim_global
        c_dim = c_dim_local
        # c_dim = c_dim_global
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.z_resolution = z_resolution
        self.normalize = normalize
        self.positional_encoding = positional_encoding
        if positional_encoding:
            dir_dim_in = 27
        else:
            dir_dim_in = 3

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.fc_p = nn.Linear(3, hidden_size)
        self.fc_dir = nn.Sequential(nn.Linear(dir_dim_in, hidden_size), nn.ReLU())
        self.fc_ori_dir = nn.Linear(hidden_size*2, hidden_size)


        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, z_resolution)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

        self.local_layer = nn.Sequential(
            nn.Linear(c_dim_local + 1, 1), nn.ReLU()
        )

    def sample_plane_feature(self, p, c):
        xy = p / 256  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c


    def forward(self,camera_ori_world, points_xy, ray_dir, c_global, c_local):
        batch_size, N, D = camera_ori_world.size()

        if self.c_dim_local != 0:
            c_local = self.sample_plane_feature(points_xy, c_local)
            c_local = c_local.transpose(1, 2)

        if self.normalize:
            c_local = F.normalize(c_local, dim=-1)
            # c_global = F.normalize(c_global, dim=-1)

        c = c_local


        if self.positional_encoding:
            ray_dir = F.normalize(ray_dir, dim=-1)
            ray_dir = encode_position(ray_dir, 4)


        ori_feature = self.fc_p(camera_ori_world)
        dir_feature = self.fc_dir(ray_dir)
        net = torch.cat([ori_feature, dir_feature], dim=-1)
        net = self.fc_ori_dir(net)


        for i in range(self.n_blocks):
            if self.c_dim != 0 and i==0:
                net = net + self.fc_c[i](c)
            # print('###########', self.fc_c[i](c).shape, net.shape) #[64, 127, 256]
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        # print('$$$$$$$$$$$$', out.shape)  #(batch, subsample, z_resolution)

        return out
class LocalDecoder_CBatchNorm(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=2, c_dim_local=128, c_dim_global=128, z_resolution = 32,
                 hidden_size=256, n_blocks=5, leaky=False, legacy=False, sample_mode='bilinear', padding=0.1, normalize=False,positional_encoding=False):
        super().__init__()
        ################################3
        self.fc_global = nn.Linear(256, c_dim_global)
        ####################################
        self.c_dim_local = c_dim_local
        self.c_dim_global = c_dim_global
        # c_dim = c_dim_local + c_dim_global
        c_dim = c_dim_local
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.z_resolution = z_resolution
        self.normalize = normalize
        self.positional_encoding = positional_encoding
        if positional_encoding:
            geo_in_dim = c_dim_local + 3 + 3*(2*4 + 1)
        else:
            geo_in_dim = c_dim_local + 3 + 3


        self.fc_geo1 = ResnetBlockFC(geo_in_dim, hidden_size)
        self.fc_geo2 = ResnetBlockFC(hidden_size, hidden_size)
        self.fc_geo3 = ResnetBlockFC(hidden_size, hidden_size)

        c_dim = 256

        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)


        # self.fc_out = nn.Linear(hidden_size, z_resolution)
        self.fc_out = nn.Conv1d(hidden_size, z_resolution, 1)

        self.sample_mode = sample_mode
        self.padding = padding

        self.local_layer = nn.Sequential(
            nn.Linear(c_dim_local + 1, 1), nn.ReLU(),

        )
        self.fc_test = nn.Linear(38, 21)

    # def sample_plane_feature(self, p, c):
    #     xy = normalize_coordinate(p.clone(), padding=self.padding)  # normalize to the range of (0, 1)
    #     xy = xy[:, :, None].float()
    #     vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
    #     c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
    #     return c

    # def sample_plane_feature2(self, p, c):
    #     p = p.unsqueeze(2)  # [B, N, 1, 2]
    #     samples = F.grid_sample(c, p, align_corners=True).squeeze(-1)  # [B, C, N, 1]
    #     return samples  # [B, C, N]
    def sample_plane_feature(self, xy, c):
        # xy = p / 256  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def forward(self, camera_ori_world, points_xy, ray_dir, c_global, c_local):

        if self.c_dim_local != 0:
            c_local = self.sample_plane_feature(points_xy, c_local)
            c_local = c_local.transpose(1, 2)

        if self.normalize:
            c_local = F.normalize(c_local, dim=-1)
            # c_global = F.normalize(c_global, dim=-1)

        if self.positional_encoding:
            ray_dir = F.normalize(ray_dir, dim=-1)
            ray_dir = encode_position(ray_dir, 4)

        net = torch.cat([camera_ori_world, ray_dir, c_local], dim=-1)
        net = self.fc_geo1(net)
        net = self.fc_geo2(net)
        net = self.fc_geo3(net)

#################################################333
        c = c_global
        net = net.transpose(1, 2)
        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.transpose(1, 2)

        return out



class PatchLocalDecoder(nn.Module):
    ''' Decoder adapted for crop training.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        local_coord (bool): whether to use local coordinate
        unit_size (float): defined voxel unit size for local system
        pos_encoding (str): method for the positional encoding, linear|sin_cos
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]

    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, leaky=False, n_blocks=5, sample_mode='bilinear', local_coord=False,
                 pos_encoding='linear', unit_size=0.1, padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        # self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

        if local_coord:
            self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
        else:
            self.map2local = None

        if pos_encoding == 'sin_cos':
            self.fc_p = nn.Linear(60, hidden_size)
        else:
            self.fc_p = nn.Linear(dim, hidden_size)

    def sample_feature(self, xy, c, fea_type='2d'):
        if fea_type == '2d':
            xy = xy[:, :, None].float()
            vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
            c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        else:
            xy = xy[:, :, None, None].float()
            vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
            c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(
                -1).squeeze(-1)
        return c

    def forward(self, p, c_plane, **kwargs):
        p_n = p['p_n']
        p = p['p']

        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_feature(p_n['grid'], c_plane['grid'], fea_type='3d')
            if 'xz' in plane_type:
                c += self.sample_feature(p_n['xz'], c_plane['xz'])
            if 'xy' in plane_type:
                c += self.sample_feature(p_n['xy'], c_plane['xy'])
            if 'yz' in plane_type:
                c += self.sample_feature(p_n['yz'], c_plane['yz'])
            c = c.transpose(1, 2)

        p = p.float()
        if self.map2local:
            p = self.map2local(p)

        net = self.fc_p(p)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


class LocalPointDecoder(nn.Module):
    ''' Decoder for PointConv Baseline.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of blocks ResNetBlockFC layers
        sample_mode (str): sampling mode  for points
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, leaky=False, n_blocks=5, sample_mode='gaussian', **kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        if sample_mode == 'gaussian':
            self.var = kwargs['gaussian_val'] ** 2

    def sample_point_feature(self, q, p, fea):
        # q: B x M x 3
        # p: B x N x 3
        # fea: B x N x c_dim
        # p, fea = c
        if self.sample_mode == 'gaussian':
            # distance betweeen each query point to the point cloud
            dist = -((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3) + 10e-6) ** 2
            weight = (dist / self.var).exp()  # Guassian kernel
        else:
            weight = 1 / ((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3) + 10e-6)

        # weight normalization
        weight = weight / weight.sum(dim=2).unsqueeze(-1)

        c_out = weight @ fea  # B x M x c_dim

        return c_out

    def forward(self, p, c, **kwargs):
        n_points = p.shape[1]

        if n_points >= 30000:
            pp, fea = c
            c_list = []
            for p_split in torch.split(p, 10000, dim=1):
                if self.c_dim != 0:
                    c_list.append(self.sample_point_feature(p_split, pp, fea))
            c = torch.cat(c_list, dim=1)

        else:
            if self.c_dim != 0:
                pp, fea = c
                c = self.sample_point_feature(p, pp, fea)

        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out
