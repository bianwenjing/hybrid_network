import torch
import numpy as np
import os
import trimesh
from plyfile import PlyData, PlyElement

def write_ply(save_path, points, text=True):
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)
def origin_to_world(n_points, camera_mat, world_mat, scale_mat, invert=True):
    ''' Transforms origin (camera location) to world coordinates.

    Args:
        n_points (int): how often the transformed origin is repeated in the
            form (batch_size, n_points, 3)
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert the matrices (default: true)
    '''
    batch_size = camera_mat.shape[0]
    device = camera_mat.device

    # Create origin in homogen coordinates
    p = torch.zeros(batch_size, 4, n_points).to(device)
    p[:, -1] = 1.

    # Invert matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Apply transformation
    p_world = scale_mat @ world_mat @ camera_mat @ p

    # Transform points back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)
    return p_world
def b_inv(b_mat):
    ''' Performs batch matrix inversion.

    Arguments:
        b_mat: the batch of matrices that should be inverted
    '''

    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    # b_inv, _ = torch.gesv(eye, b_mat)
    b_inv, _ = torch.solve(eye, b_mat)
    return b_inv

def cam_to_world(pts, camera_mat, world_mat, scale_mat, invert=True):
    ''' Transforms origin (camera location) to world coordinates.

    Args:
        n_points (int): how often the transformed origin is repeated in the
            form (batch_size, n_points, 3)
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert the matrices (default: true)
    '''
    R = world_mat[:, :3, :3]
    t = world_mat[:, :3, 3:]
    pts = pts - t.squeeze()

    pts = pts @ b_inv(R.transpose(0, 1))
    # batch_size = camera_mat.shape[0]
    # device = camera_mat.device
    #
    # # # Create origin in homogen coordinates
    # # p = torch.zeros(batch_size, 4, n_points).to(device)
    # # p[:, -1] = 1.
    #
    # # Invert matrices
    # if invert:
    #     camera_mat = torch.inverse(camera_mat)
    #     world_mat = torch.inverse(world_mat)
    #     scale_mat = torch.inverse(scale_mat)
    #
    # # Apply transformation
    # p_world = scale_mat @ world_mat  @ p
    #
    # # Transform points back to 3D coordinates
    # p_world = p_world[:, :3].permute(0, 2, 1)
    return pts

def get_rays(points_xy, focal, device=None):
    # W = 256
    # H = 256
    # i = points_xy[:, :, 0] * W
    # j = points_xy[:, :, 1] * H
    points_xy = (points_xy-0.5)*2
    i = points_xy[:, :, 0]
    j = points_xy[:, :, 1]

    # dirs = torch.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)], -1)
    dirs = torch.stack([i  / focal, -j / focal, -torch.ones_like(i)], dim=-1)

    return dirs



folder = '/home/wenjing/Downloads/DVR/03001627/'
camera_file = os.path.join(folder, 'cameras.npz')
mesh_file = os.path.join(folder, '7f4f73ad1b3f882ba14472becb07b261_0_in.off')

camera_dict = np.load(camera_file)
mesh = trimesh.load(mesh_file, process=False)

idx = 2
world_mat = torch.tensor(camera_dict['world_mat_%d' % idx].astype(np.float32)).unsqueeze(0)
camera_mat = torch.tensor(camera_dict['camera_mat_%d' % idx].astype(np.float32)).unsqueeze(0)
scale_mat = torch.tensor(camera_dict.get(
    'scale_mat_%d' % idx, np.eye(4)).astype(np.float32)).unsqueeze(0)
n_points = 20
N_sam = 12
points_xy = torch.rand(1, n_points, 2)
focal = camera_mat[0, 0, 0]


ray_dir = get_rays(points_xy, focal)
t_vals = torch.linspace(0, 1.75, N_sam).view(1, N_sam).expand(n_points, N_sam).unsqueeze(0)
# print(ray_dir.shape, t_vals.shape)
pts = ray_dir.unsqueeze(2) * t_vals.unsqueeze(3)
pts = pts.reshape(-1, 3)
# print(pts.shape)

# camera_ori_world = origin_to_world(
#             n_points, camera_mat, world_mat, scale_mat)
camera_ori_world = cam_to_world(
            pts, camera_mat, world_mat, scale_mat)

points_dir = os.path.join(folder, 'ray.ply')
write_ply(points_dir, camera_ori_world[0])