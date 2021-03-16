import trimesh
import torch
import numpy as np
import os
from im2mesh.utils.libmesh import check_mesh_contains
from plyfile import PlyData, PlyElement

def write_ply(save_path, points, text=True):
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)

def b_inv(b_mat):
    ''' Performs batch matrix inversion.

    Arguments:
        b_mat: the batch of matrices that should be inverted
    '''

    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    # b_inv, _ = torch.gesv(eye, b_mat)
    b_inv, _ = torch.solve(eye, b_mat)
    return b_inv

def get_rays3(H, W, focal, world_mat):
    R = world_mat[:3, :3]
    t = world_mat[:3, 3:]


    y, x = torch.meshgrid(torch.linspace(-1, 1, H, dtype=torch.float32), torch.linspace(-1, 1, W, dtype=torch.float32))  # (H, W)
    dirs_x = 0.5* x / focal  # (H, W)
    dirs_y = -0.5*y / focal  # (H, W)
    # y, x = torch.meshgrid(torch.linspace(0, 136, H, dtype=torch.float32),
    #                       torch.linspace(0, 136, W, dtype=torch.float32))  # (H, W)
    # dirs_x = (x - 0.5 * 137) / focal  # (H, W)
    # dirs_y = -(y - 0.5 * 137) / focal  # (H, W)
    dirs_z = torch.ones(H, W, dtype=torch.float32)  # (H, W)
    ray_dir_cam = torch.stack([dirs_x, dirs_y, torch.ones_like(dirs_x)], dim=-1)  # (H, W, 3)
    N_sam = 64
    t_vals = torch.linspace(0, 2.4, N_sam)
    t_vals_noisy = t_vals.view(1, 1, N_sam).expand(H, W, N_sam)
    # print(ray_dir_cam.shape, t_vals_noisy.shape)
    pts = ray_dir_cam.unsqueeze(2) * t_vals_noisy.unsqueeze(3)
    # print(pts.shape, t.shape)
    pts = pts - t.squeeze()

    pts = pts @ b_inv(R.transpose(0, 1))

    # rays_d = torch.matmul(c2w[:3, :3].view(1, 1, 3, 3),
    #                              ray_dir_cam.unsqueeze(3)).squeeze(3)  # (1, 1, 3, 3) * (H, W, 3, 1) -> (H, W, 3)
    # rays_o = c2w[:3, 3].expand(rays_d.shape)  # the translation vector (3, )

    return pts

folder = '/home/wenjing/Downloads/DVR/03001627/'
camera_file = os.path.join(folder, 'cameras.npz')
mesh_file = os.path.join(folder, '7f4f73ad1b3f882ba14472becb07b261_0_in.off')

camera_dict = np.load(camera_file)
mesh = trimesh.load(mesh_file, process=False)

idx = 6
world_mat = torch.tensor(camera_dict['world_mat_%d' % idx].astype(np.float32))
camera_mat = torch.tensor(camera_dict['camera_mat_%d' % idx].astype(np.float32))
scale_mat = torch.tensor(camera_dict.get(
    'scale_mat_%d' % idx, np.eye(4)).astype(np.float32))
focal = camera_mat[0][0]
print(scale_mat)

H = 13
W = 13
pts = get_rays3(H, W, focal, world_mat)
pts = pts.reshape(-1, 3).numpy()

points_dir = os.path.join(folder, 'rays.ply')
write_ply(points_dir, pts)