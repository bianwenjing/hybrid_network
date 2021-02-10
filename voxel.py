import os
import glob
import random
from PIL import Image
import numpy as np
import trimesh
from im2mesh.data.core import Field
from im2mesh.utils import binvox_rw
import torch
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision.utils import save_image
import im2mesh.common as common
from im2mesh import data
from im2mesh.utils.libmesh import check_mesh_contains

class RayField2_cam(Field):
    def __init__(self, file_name, transform=None, z_resolution=32, with_transforms=False, unpackbits=False, img_folder=None):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms
        self.unpackbits = unpackbits
        self.z_resolution = z_resolution
        self.folder_name = img_folder

    def get_rays(self, H, W, focal, c2w):
        i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                              torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
        i = i.t() #transpose
        j = j.t()

        dirs = torch.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                           -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3, -1].expand(rays_d.shape)

        i = torch.reshape(i, [-1, 1])
        j = torch.reshape(j, [-1, 1])
        points_xy = torch.cat([i, j], dim=1)

        return rays_o, rays_d, points_xy

    def ndc_rays(self, H, W, focal, near, rays_o, rays_d):
        # Shift ray origins to near plane
        t = -(near + rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d

        # Projection
        o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
        o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
        o2 = 1. + 2. * near / rays_o[..., 2]

        d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
        d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
        d2 = -2. * near / rays_o[..., 2]

        rays_o = torch.stack([o0, o1, o2], -1)
        rays_d = torch.stack([d0, d1, d2], -1)

        return rays_o, rays_d

    def load(self, model_path, idx, category, idx_img):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        in_path = os.path.join(model_path, self.file_name)  #load mesh
        mesh = trimesh.load(in_path, process=False)
        H = 137
        W = 137
        ndc = False
        N_samples = self.z_resolution

        folder = os.path.join(model_path, self.folder_name)
        camera_file = os.path.join(folder, 'cameras.npz')
        camera_dict = np.load(camera_file)
        # world_mat = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
        # camera_mat = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)
        world_mat = torch.tensor(camera_dict['world_mat_%d' % idx_img])
        camera_mat = torch.tensor(camera_dict['camera_mat_%d' % idx_img])
        focal = camera_mat[0][0]

        rays_o, rays_d, points_xy = self.get_rays(H, W, focal, world_mat)
        # print('$$$$$$$$$$$$', rays_d.shape, rays_o.shape)
        if ndc:
            # for forward facing scenes
            rays_o, rays_d = self.ndc_rays(H, W, focal, 1., rays_o, rays_d)

        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        t_vals = torch.linspace(0., 1., steps=N_samples)
        near = 0
        far = -1
        z_vals = near * (1. - t_vals) + far * (t_vals)
        N_rays = rays_d.shape[0]
        z_vals = z_vals.expand([N_rays, N_samples])
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        pts = pts.reshape(-1, 3).numpy()
        points_xy = points_xy.numpy()
        occupancies = check_mesh_contains(mesh, pts)

        if points_xy.dtype == np.float16:
            points_xy = points_xy.astype(np.float32)
            points_xy += 1e-4 * np.random.randn(*points_xy.shape)
        else:
            points_xy = points_xy.astype(np.float32)

        occupancies = occupancies.reshape(-1, N_samples)
        occupancies_r = occupancies.astype(np.float32)
        data = {
            None: points_xy,
            'occ': occupancies_r,
        }
        # if self.with_transforms:
        #     data['loc'] = points_dict['loc'].astype(np.float32)
        #     data['scale'] = points_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data
points_transform = data.SubsamplePoints(128)
file_name = 'mesh.off'
a = RayField2_cam(file_name,transform=points_transform, img_folder='img_choy2016')
model_path = '/home/wenjing/Downloads/ShapeNet/02828884/c8802eaffc7e595b2dc11eeca04f912e'
a.load(model_path, 1, 1, 23)
# def make_2d_grid(bb_min, bb_max, shape):
#     ''' Makes a 3D grid.
#
#     Args:
#         bb_min (tuple): bounding box minimum
#         bb_max (tuple): bounding box maximum
#         shape (tuple): output shape
#     '''
#     size = shape[0] * shape[1]
#
#     pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
#     pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
#
#
#     pxs = pxs.view(-1, 1).expand(*shape).contiguous().view(size)
#     pys = pys.view(1, -1).expand(*shape).contiguous().view(size)
#
#     p = torch.stack([pxs, pys], dim=1)
#
#     return p
# num_x = 8
# num_yz = 3
# points_yz = make_2d_grid(
#                 (-0.5,)*2, (0.5,)*2, (num_yz,)*2
#             )
#
# points_x = np.random.rand(8, 1)
# points_x = np.repeat(points_x, num_yz**2)
# points_x = np.expand_dims(points_x, axis=1)
# points_yz = np.expand_dims(points_yz, axis=0)
# points_yz = np.repeat(points_yz, num_x, axis=0).reshape(num_x*num_yz**2, 2)
#
#
# points_uniform = np.concatenate([points_x, points_yz], axis=1)
# # points_uniform = points_uniform.reshape(num_x, num_yz, num_yz,  3)
# print(points_uniform)

# values = np.random.rand(4, 4, 4)
# end_pad1 =  np.zeros((values.shape[0], 1, values.shape[2]))
# end_pad2 =  np.zeros((values.shape[0], values.shape[1]+1, 1))
# values = np.concatenate((values, end_pad1), axis=1)
# values = np.concatenate((values, end_pad2), axis=2)
# print(values)

# # data = np.load('/media/wenjing/Data21/ShapeNet.build/04090263/4_points/ffe20b3b5f34fbd0cbc6ff5546f4ec42.npz')
# data = np.load('/media/wenjing/Data21/ShapeNet.build/02828884/4_points/46b3e857b7faab0117f92404d4be5d8.npz')
# # data = np.load('/media/wenjing/Data21/ShapeNet.build/03001627/4_points/fffda9f09223a21118ff2740a556cc3.npz')
# lst = data.files
# # for item in lst:
# #     print(item)
# #     print(data[item], data[item].shape)
# points_x = data['points_x']
# len = points_x.shape[0]*65*65
# occupancies = np.unpackbits(data['occupancies'])[:len]
# occupancies = occupancies.reshape(100, 65, 65)
# print(occupancies)


# points_x = np.random.rand(4)
# points_y = np.random.rand(4)
# points_z = np.expand_dims(np.linspace(0, 1, num=5),axis=1)
# print(points_z.shape)
# points_x = np.expand_dims(np.random.choice(points_x, 12, replace=True),axis=1)
#
# points_y = np.expand_dims(np.random.choice(points_y, 12, replace=True), axis=1)
# points = np.concatenate([points_x, points_y, points_z], axis=1)
# print(points.shape)
# data = np.load('/home/wenjing/storage/raw/ShapeNet.build/03001627/4_points/fffda9f09223a21118ff2740a556cc3.npz')
# voxels = np.unpackbits(data['occupancies']).reshape(2500, 128)
# points_xy = np.random.rand(16,2)
# points_xy = np.repeat(points_xy,5,axis=0)
# # print(points_xy.shape, points_xy)
# points_z = np.linspace(0, 1, num=5)
# points_z = np.expand_dims(points_z, axis=1)
# points_z = np.repeat(points_z, 16, axis=1).transpose(1,0)
# points_z = np.reshape(points_z, (-1,1))
# print(points_z.shape)
# points = np.concatenate([points_xy, points_z], axis=1)
# print(points.shape, points)
# file_path = '/home/wenjing/Downloads/ShapeNet/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/model.binvox'
# with open(file_path, 'rb') as f:
#     voxels = binvox_rw.read_as_3d_array(f)
# print(voxels)
# voxels = voxels.data.astype(np.float32)
# voxels[:,15,15] = 0
# print(voxels.shape, voxels)
# x = np.linspace(-0.5, 0.5,num=voxels.shape[0])
# y = np.linspace(-0.5, 0.5,num=voxels.shape[1])
# points_xy = np.concatenate([x,y], axis=0)
# print(points_xy.shape, points_xy)


# points_xy = []
# for i in np.linspace(-0.5, 0.5,num=voxels.shape[0]):
#     for j in np.linspace(-0.5, 0.5,num=voxels.shape[1]):
#         points_xy.append([i,j])
#
# # points_xy = np.array(points_xy)
# points_xy = np.array(points_xy).astype(np.float32)
# # points_xy += 1e-4 * np.random.randn(*points_xy.shape)
# # print(points_xy.shape)

# voxels = voxels.transpose(2,1,0)
# voxels = voxels.reshape(1024, 32)
# # voxels_2 = voxels.reshape(1024,32)
# # voxels_2 = voxels_2[0:300]
# # voxels_2 = np.expand_dims(voxels_2, axis=0)
#
# # print(voxels_2.shape)
# #
# # # print(voxels.shape)
# x = np.array([[[1,2],[3,4]],[[5,6], [7,8]]])
# print(x.shape)
# x=x.transpose(0,1,2)
# # x[:,1,1] = 0
# # x = x.reshape(4,2)
# print(x)


# # Create plot
# fig = plt.figure()
# ax = fig.gca(projection=Axes3D.name)
# voxels = voxels.transpose(2, 0, 1)
# voxels = voxels.transpose(1, 0, 2)
# # voxels_2 = voxels_2.transpose(2, 0, 1)
# ax.voxels(voxels, edgecolor='k')
# # ax.voxels(voxels_2, edgecolor='r')
# ax.set_xlabel('Z')
# ax.set_ylabel('X')
# ax.set_zlabel('Y')
# # ax.view_init(elev=30, azim=45)
# plt.show()
