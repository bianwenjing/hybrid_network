import os
import glob
import random
from PIL import Image
import numpy as np
import trimesh
from im2mesh.data.core import Field
from im2mesh.utils import binvox_rw
import torch
from im2mesh.utils.libmesh import check_mesh_contains

class IndexField(Field):
    ''' Basic index field.'''
    def load(self, model_path, idx, category, img_index):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True


class CategoryField(Field):
    ''' Basic category field.'''
    def load(self, model_path, idx, category, img_index):
        ''' Loads the category field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return category

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True


class ImagesField(Field):
    ''' Image Field.

    It is the field used for loading images.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
        random_view (bool): whether a random view should be used
        with_camera (bool): whether camera data should be provided
    '''
    def __init__(self, folder_name, transform=None,
                 extension='jpg', random_view=True, with_camera=False):
        self.folder_name = folder_name
        self.transform = transform
        self.extension = extension
        self.random_view = random_view
        self.with_camera = with_camera

    def load(self, model_path, idx, category, img_index):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, '*.%s' % self.extension))
        # files.sort()
        if self.random_view:
            # idx_img = random.randint(0, len(files)-1)
            idx_img = img_index
        else:
            idx_img = 0
        filename = files[idx_img]

        image = Image.open(filename).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)



        data = {
            None: image
        }

        if self.with_camera:
            camera_file = os.path.join(folder, 'cameras.npz')
            camera_dict = np.load(camera_file)
            Rt = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
            K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)
            data['world_mat'] = Rt
            data['camera_mat'] = K

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.folder_name in files)
        # TODO: check camera
        return complete


# 3D Fields
class PointsField(Field):
    ''' Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided

    '''
    def __init__(self, file_name, transform=None, with_transforms=False, unpackbits=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms
        self.unpackbits = unpackbits

    def load(self, model_path, idx, category, img_index):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
        else:
            points = points.astype(np.float32)

        occupancies = points_dict['occupancies']

        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)
        data = {
            None: points,
            'occ': occupancies,
        }

        if self.with_transforms:
            data['loc'] = points_dict['loc'].astype(np.float32)
            data['scale'] = points_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data
class RayField2_cam(Field):
    def __init__(self, file_name, transform=None, z_resolution=32, with_transforms=False, img_folder=None):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms
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
        # points_dict = np.load(file_path)
        # points_xy = points_dict['points_xy']
        # Break symmetry if given in float16:

        # points_xy = (points_xy - 0.5) * 1.1
        if points_xy.dtype == np.float16:
            points_xy = points_xy.astype(np.float32)
            points_xy += 1e-4 * np.random.randn(*points_xy.shape)
        else:
            points_xy = points_xy.astype(np.float32)

        occupancies = occupancies.reshape(-1, N_samples)
        occupancies = occupancies.astype(np.float32)
        data = {
            None: points_xy,
            'occ': occupancies,
        }
        # if self.with_transforms:
        #     data['loc'] = points_dict['loc'].astype(np.float32)
        #     data['scale'] = points_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)
        return data

class RayField2(Field):
    def __init__(self, file_name, transform=None, z_resolution=32, with_transforms=False, unpackbits=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms
        self.unpackbits = unpackbits
        self.z_resolution = z_resolution

    def load(self, model_path, idx, category, img_index):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        points_dict = np.load(file_path)
        points_xy = points_dict['points_xy']
        # Break symmetry if given in float16:

        # points_xy = (points_xy - 0.5) * 1.1

        if points_xy.dtype == np.float16:
            points_xy = points_xy.astype(np.float32)
            points_xy += 1e-4 * np.random.randn(*points_xy.shape)
        else:
            points_xy = points_xy.astype(np.float32)

        occupancies = points_dict['occupancies']

        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)

        occupancies = occupancies.reshape(2500, 128)
        interval = int(128/self.z_resolution)
        index = [i*interval for i in range(self.z_resolution)]
        occupancies_r = occupancies[:, index]
        occupancies_r = occupancies_r.astype(np.float32)
        data = {
            None: points_xy,
            'occ': occupancies_r,
        }
        if self.with_transforms:
            data['loc'] = points_dict['loc'].astype(np.float32)
            data['scale'] = points_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

class RayField1(Field):
    def __init__(self, file_name, transform=None, yz_resolution=32, with_transforms=False, unpackbits=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms
        self.unpackbits = unpackbits
        self.yz_resolution = yz_resolution

    def load(self, model_path, idx, category, img_index):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        points_dict = np.load(file_path)
        points_x = points_dict['points_x']
        # Break symmetry if given in float16:

        if points_x.dtype == np.float16:
            points_xy = points_x.astype(np.float32)
            points_xy += 1e-4 * np.random.randn(*points_xy.shape)
        else:
            points_xy = points_x.astype(np.float32)

        occupancies = points_dict['occupancies']

        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:422500]

        occupancies = occupancies.reshape(100, 65, 65)
        interval = int(64/self.yz_resolution)
        index = [i*interval for i in range(self.yz_resolution)]
        # index = 64
        occupancies_r = occupancies[:, :, index][:, index, :]
        occupancies_r = occupancies_r.astype(np.float32)
        data = {
            None: points_xy,
            'occ': occupancies_r,
        }
        if self.with_transforms:
            data['loc'] = points_dict['loc'].astype(np.float32)
            data['scale'] = points_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

class RayField(Field):
    ''' Voxel field class.

    It provides the class used for voxel-based data.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    '''
    def __init__(self, file_name, transform=None, with_transforms=False,):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms

    def load(self, model_path, idx, category, img_index):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        with open(file_path, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f)
        voxels = voxels.data.astype(np.float32)
        points_xy = []
        for i in np.linspace(-0.5, 0.5, num=voxels.shape[0]):
            for j in np.linspace(-0.5, 0.5, num=voxels.shape[1]):
                points_xy.append([i, j])
        points_xy = np.array(points_xy).astype(np.float32)
        points_xy += 1e-4 * np.random.randn(*points_xy.shape)

        voxels = voxels.transpose(2, 1, 0)
        occupancies_z = voxels.reshape(1024, 32)
        data = {
            None: points_xy,
            'occ': occupancies_z,
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

class VoxelsField(Field):
    ''' Voxel field class.

    It provides the class used for voxel-based data.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category, img_index):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        with open(file_path, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f)
        voxels = voxels.data.astype(np.float32)

        if self.transform is not None:
            voxels = self.transform(voxels)

        return voxels

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


class PointCloudField(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        with_transforms (bool): whether scaling and rotation dat should be
            provided
    '''
    def __init__(self, file_name, transform=None, with_transforms=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms

    def load(self, model_path, idx, category, img_index):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        data = {
            None: points,
            'normals': normals,
        }

        if self.with_transforms:
            data['loc'] = pointcloud_dict['loc'].astype(np.float32)
            data['scale'] = pointcloud_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


# NOTE: this will produce variable length output.
# You need to specify collate_fn to make it work with a data laoder
class MeshField(Field):
    ''' Mesh field.

    It provides the field used for mesh data. Note that, depending on the
    dataset, it produces variable length output, so that you need to specify
    collate_fn to make it work with a data loader.

    Args:
        file_name (str): file name
        transform (list): list of transforms applied to data points
    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category, img_index):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        mesh = trimesh.load(file_path, process=False)
        if self.transform is not None:
            mesh = self.transform(mesh)

        data = {
            'verts': mesh.vertices,
            'faces': mesh.faces,
        }

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete
