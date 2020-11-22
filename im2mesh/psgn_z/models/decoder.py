import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    r''' Simple decoder for the Point Set Generation Network.

    The simple decoder consists of 4 fully-connected layers, resulting in an
    output of 3D coordinates for a fixed number of points.

    Args:
        dim (int): The output dimension of the points (e.g. 3)
        c_dim (int): dimension of the input vector
        n_points (int): number of output points
    '''
    def __init__(self, dim=3, z_dim=128, c_dim=128, n_points=1024):
        super().__init__()
        # Attributes
        self.dim = dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.n_points = n_points

        # Submodules
        self.actvn = F.relu
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, 512)
        if not c_dim == 0:
            self.fc_0 = nn.Linear(c_dim, 512)
        self.fc_1 = nn.Linear(512, 512)
        self.fc_2 = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, dim*n_points)

    def forward(self, z, c):
        if self.z_dim != 0:
            net = self.fc_z(z)
            batch_size = z.size(0)
        if self.c_dim != 0:
            net = self.fc_0(c)
            batch_size = c.size(0)

        net = self.fc_1(self.actvn(net))
        net = self.fc_2(self.actvn(net))
        points = self.fc_out(self.actvn(net))
        points = points.view(batch_size, self.n_points, self.dim)

        return points
