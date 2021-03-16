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

def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, 136, W),
                          torch.linspace(0,136, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()  # transpose
    j = j.t()

    dirs = torch.stack([(i - 137 * .5) / focal, -(j - 137 * .5) / focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    i = torch.reshape(i, [-1, 1])
    j = torch.reshape(j, [-1, 1])
    points_xy = torch.cat([i, j], dim=1)

    return rays_o, rays_d
def get_rays2(H, W, focal, c2w):
    y, x = torch.meshgrid(torch.linspace(0, 136, H, dtype=torch.float32), torch.linspace(0, 136, W, dtype=torch.float32))  # (H, W)
    dirs_x = (x - 0.5 * 137) / focal  # (H, W)
    dirs_y = -(y - 0.5 * 137) / focal  # (H, W)
    dirs_z = -torch.ones(H, W, dtype=torch.float32)  # (H, W)
    ray_dir_cam = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # (H, W, 3)
    rays_d = torch.matmul(c2w[:3, :3].view(1, 1, 3, 3),
                                 ray_dir_cam.unsqueeze(3)).squeeze(3)  # (1, 1, 3, 3) * (H, W, 3, 1) -> (H, W, 3)
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # the translation vector (3, )

    return rays_o, rays_d
def get_rays3(H, W, focal, world_mat):
    R = world_mat[:, :3]
    t = world_mat[:, 3:]

    y, x = torch.meshgrid(torch.linspace(0, 136, H, dtype=torch.float32), torch.linspace(0, 136, W, dtype=torch.float32))  # (H, W)
    dirs_x = (x - 0.5 * 137) / focal  # (H, W)
    dirs_y = -(y - 0.5 * 137) / focal  # (H, W)
    dirs_z = torch.ones(H, W, dtype=torch.float32)  # (H, W)
    ray_dir_cam = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # (H, W, 3)
    N_sam = 12
    t_vals = torch.linspace(0, 1.75, N_sam)
    t_vals_noisy = t_vals.view(1, 1, 12).expand(H, W, N_sam)
    # print(ray_dir_cam.shape, t_vals_noisy.shape)
    pts = ray_dir_cam.unsqueeze(2) * t_vals_noisy.unsqueeze(3)
    # print(pts.shape, t.shape)
    pts = pts - t.squeeze()

    pts = pts @ b_inv(R.transpose(0, 1))

    # rays_d = torch.matmul(c2w[:3, :3].view(1, 1, 3, 3),
    #                              ray_dir_cam.unsqueeze(3)).squeeze(3)  # (1, 1, 3, 3) * (H, W, 3, 1) -> (H, W, 3)
    # rays_o = c2w[:3, 3].expand(rays_d.shape)  # the translation vector (3, )

    return pts
# mesh_path = '/home/wenjing/Desktop/test_mesh/2_watertight.off'
# mesh = trimesh.load(mesh_path, process=False)
H = 13
W = 13
N_samples = 12
idx_img = 3
folder = '/home/wenjing/Desktop/test_mesh'
camera_file = os.path.join(folder, 'img_choy2016/cameras.npz')
camera_dict = np.load(camera_file)
# world_mat = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
# camera_mat = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)

world_mat = torch.tensor(camera_dict['world_mat_%d' % idx_img].astype(np.float32))
camera_mat = torch.tensor(camera_dict['camera_mat_%d' % idx_img].astype(np.float32))
focal = camera_mat[0][0]
print('f', world_mat.shape, world_mat[:, 3:])

R = world_mat[:, :3]
t = world_mat[:, 3:]
# Rt = fix_Rt_camera(Rt, loc, scale)
# K = fix_K_camera(K, img_size=137.)
# c2w = b_inv(R.transpose(0, 1))
# t = - c2w @ t
# t = - t.transpose(0, 1) @ c2w

# c2w = torch.cat([R, t], dim=1)
# points_out = points - t.transpose(1, 2)
# points_out = points_out @ b_inv(R.transpose(1, 2))
# rays_o, rays_d = get_rays(H, W, focal, c2w)
# t_vals = torch.linspace(0., 1., steps=N_samples)
# near = 0
# far = 1
# z_vals = near * (1. - t_vals) + far * (t_vals)
# N_rays = rays_d.shape[0]
# z_vals = z_vals.expand([N_rays, N_samples])
# pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


pts = get_rays3(H, W, focal, world_mat)
pts = pts.reshape(-1, 3).numpy()

points_dir = os.path.join(folder, 'rays_.ply')
write_ply(points_dir, pts)

# # points_xy = points_xy.numpy()
# # occupancies = check_mesh_contains(mesh, pts)
# pointcloud_path = os.path.join(folder, 'pointcloud.npz')
# pointcloud = np.load(pointcloud_path)
# lst = pointcloud.files
# # for item in lst:
# #     print(item)
# scale = torch.tensor(pointcloud['scale'].astype(np.float32))
# loc = torch.tensor(pointcloud['loc'].astype(np.float32))
#
# points = torch.tensor(pointcloud['points'].astype(np.float32))
#
# points_dir = os.path.join(folder, 'pointcloud.ply')
# # write_ply(points_dir, points)
# points_rotate = points @ R.transpose(0, 1)
# points_rotate_dir = os.path.join(folder, 'pointcloud_rotate.ply')
# # write_ply(points_rotate_dir, points_rotate)
# # print(points_rotate.shape, t.shape, t)
# points_translate = points_rotate + t.transpose(0, 1)
# points_trans_dir = os.path.join(folder, 'pointcloud_trans.ply')
# # write_ply(points_trans_dir, points_translate)
#
# # points_trans_rot = (points + t.transpose(0, 1)) @ R.transpose(0, 1)
# # points_trans_rot_dir = os.path.join(folder, 'pointcloud_trans_rot.ply')
# # write_ply(points_trans_rot_dir, points_trans_rot)
#
# # points_trans2 = points + t.transpose(0, 1)
# # points_trans2_dir = os.path.join(folder, 'pointcloud_trans2.ply')
# # write_ply(points_trans2_dir, points_trans2)
#
# scale_mat = torch.tensor([
#         [2./137, 0, -1],
#         [0, 2./137, -1],
#         [0, 0, 1.],
#     ], dtype=camera_mat.dtype)
#
# K = scale_mat @ camera_mat
# projection = points_translate @ K.transpose(0, 1)
# p_camera = projection[..., :] / projection[..., 2:]
#
#
# # projection_path = os.path.join(folder, 'projection.ply')
# # write_ply(projection_path, projection)
# p_camera_path = os.path.join(folder, 'p_camera.ply')
# # write_ply(p_camera_path, p_camera)