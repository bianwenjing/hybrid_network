import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange
import trimesh
# from im2mesh.utils import libmsquares
from im2mesh.utils import libmcubes
from im2mesh.common import make_2d_grid
from im2mesh.utils.libsimplify import simplify_mesh
from im2mesh.utils.libmise import MISE
import time
import im2mesh.common as common
import im2mesh.common2 as common2
import torch.nn as nn
import torch.nn.functional as F

class Generator3D(object):
    '''  Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        simplify_nfaces (int): number of faces the mesh should be simplified to
        preprocessor (nn.Module): preprocessor for inputs
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
                 z_resolution=32,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1, sample=False,
                 simplify_nfaces=None,
                 preprocessor=None, camera=True):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces
        self.preprocessor = preprocessor
        self.z_resolution = z_resolution
        self.camera = camera

    def generate_mesh(self, data, return_stats=True):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get('img').to(device)
        kwargs = {}

        # Preprocess if requires
        if self.preprocessor is not None:
            t0 = time.time()
            with torch.no_grad():
                inputs = self.preprocessor(inputs)
            stats_dict['time (preprocess)'] = time.time() - t0

        # Encode inputs
        t0 = time.time()
        with torch.no_grad():
            c, c_local = self.model.encode_inputs(inputs)
        stats_dict['time (encode inputs)'] = time.time() - t0

        # z = self.model.get_z_from_prior((1,), sample=self.sample).to(device)
        ###########################################################
        mesh = self.generate_from_latent(c, c_local, data, stats_dict=stats_dict, **kwargs)
        ######################################################################

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh

    def generate_from_latent(self, c=None, c_local=None, data=None, stats_dict={}, **kwargs):
        ''' Generates mesh from latent.

        Args:
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        self.padding = 0
        box_size = 1 + self.padding
        nz = self.z_resolution
        nx = self.resolution0
        # nx = 30
        ############################
        points_xy = box_size * make_2d_grid(
            (-1.0,)*2, (1.0,)*2, (nx,)*2
        )
        points_xy = points_xy.unsqueeze(0).to(self.device)
        world_mat = data.get('img.world_mat').to(self.device)
        camera_mat = data.get('img.camera_mat').to(self.device)
        scale_mat = data.get('img.scale_mat').to(self.device)
        camera_ori_world = common2.origin_to_world(
            nx*nx, camera_mat, world_mat, scale_mat)
        focal = camera_mat[0, 0, 0]
        ray_dir = common2.get_rays(points_xy, focal, device=self.device)
        ####################################
        values = self.eval_points(camera_ori_world, points_xy, ray_dir, c, c_local, **kwargs) # (1, 4096, 64)

        value_grid = values.reshape(nx, nx, nz)
        # change to regular grid
        value_grid = self.to_regular_grid(value_grid)
        value_grid = value_grid.reshape(nx, nx, nz-1)

        value_grid = value_grid.cpu().numpy()


        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0
        # value_grid = value_grid.transpose(2, 1, 0)
        mesh = self.extract_mesh(value_grid, c, c_local, world_mat, stats_dict=stats_dict)
        return mesh
    def to_regular_grid(self, value_grid):

        value_grid = value_grid.unsqueeze(0).unsqueeze(0)
        scaled_grid = []
        for i in range(1, self.z_resolution):
            #grid sample
            scale = self.z_resolution / i
            points_xy =  scale * make_2d_grid(
            (-1.0,)*2, (1.0,)*2, (self.resolution0,)*2
        )
            points_xy = points_xy.unsqueeze(0).unsqueeze(2).to(self.device)  # [B, N, 1, 2]
            # print(points_xy.shape, value_grid.shape)
            c = F.grid_sample(value_grid[:, :, :, :, i], points_xy, padding_mode='border', align_corners=True, mode='bilinear') #(1, 1, 4096,1)
            scaled_grid.append(c)

        scaled_grid = torch.cat(scaled_grid, axis=3) # (1, z_resolution-1, num_samples)

            # ratio = self.z_resolution / i
            # x = value_grid[:, :, i].unsqueeze(0).unsqueeze(0)
            # print(x.shape)
            # x = F.interpolate(x, scale_factor=i, mode='bilinear', align_corners=True)
            # # avg = nn.AvgPool2d(ratio)
            # # x = avg(value_grid[:, :, i])
            # print('*******', x.shape)

        return scaled_grid

    def eval_points(self,camera_ori_world, points_xy, ray_dir, c=None, c_local=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        # p_split = torch.split(p, self.points_batch_size)
        # p_projected = torch.split(p_projected, self.points_batch_size)


        # for pi in p_split:

        with torch.no_grad():
            occ_hat = self.model.decode(camera_ori_world, points_xy, ray_dir, c, c_local).logits

        # occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = occ_hat.detach()

        return occ_hat

    def extract_mesh(self, occ_hat, c=None, c_local=None, world_mat=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)

        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        vertices = box_size * (vertices - 0.5)

        # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
        # mesh_pymesh = fix_pymesh(mesh_pymesh)

        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, c, c_local)
            stats_dict['time (normals)'] = time.time() - t0

        else:
            normals = None

        # # transform to world coordinate
        transformed_pred = common.transform_points_back(vertices, world_mat)
        vertices = transformed_pred.squeeze().cpu().numpy()

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c, c_local)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh

    def estimate_normals(self, vertices, c=None, c_local=None):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        c , c_local = c.unsqueeze(0), c_local.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decode(vi, c, c_local).logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, c=None, c_local=None):
        ''' Refines the predicted mesh.

        Args:
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.model.decode(face_point.unsqueeze(0), c, c_local).logits
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh
