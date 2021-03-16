import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange
import trimesh
from im2mesh.utils import libmcubes
from im2mesh.common import make_3d_grid
from im2mesh.utils.libsimplify import simplify_mesh
from im2mesh.utils.libmise import MISE
import time
import tempfile
import os
import subprocess
from im2mesh.utils.libkdtree import KDTree

from im2mesh.utils.io import export_pointcloud
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
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1, sample=False,
                 simplify_nfaces=None,
                 preprocessor=None):
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

    def generate_mesh(self, data, gt=False, return_stats=True):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
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
            c = self.model.encode_inputs(inputs)
        stats_dict['time (encode inputs)'] = time.time() - t0

        z = self.model.get_z_from_prior((1,), sample=self.sample).to(device)
        ###########################################################
        mesh, mesh2, out_points, in_points = self.generate_from_latent(z, c, gt, stats_dict=stats_dict, **kwargs)
        ######################################################################
        # return mesh, mesh2

        if return_stats:
            return mesh,mesh2,out_points, in_points, stats_dict
        else:
            return mesh, mesh2,out_points, in_points
    def max_min_dist(self, points):
        distances = []
        for i, p1 in enumerate(points):
            p2 = np.delete(points, i, 0)
            p1 = np.expand_dims(p1, axis=0)
            kdtree = KDTree(p2)
            dist, idx = kdtree.query(p1, k=1)
            distances.append(dist)
        distances = np.array(distances)
        dist_max = np.amax(distances)
        return dist_max
    def partial_mesh(self, points_psgn, z, c=None, **kwargs):
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        box_size = 0.06
        nx = 2
        upsampling_steps = 1
        points_psgn = points_psgn.squeeze(0)
        vertices = []
        faces = []
        vertices_index = 0
        if self.with_normals:
            normals = []
        else:
            normals = None
        for p in points_psgn:

            if upsampling_steps == 0:
                pointsf = box_size * make_3d_grid(
                    (-0.5,) * 3, (0.5,) * 3, (nx,) * 3
                )
                pointsf = torch.FloatTensor(pointsf).to(self.device) + p
                # print('%%%%%%%%%%%%%%', pointsf)
                values = self.eval_points(pointsf, z, c, **kwargs).cpu().numpy()
                value_grid = values.reshape(nx, nx, nx)
            else:
                mesh_extractor = MISE(
                    nx, upsampling_steps, threshold)
                points = mesh_extractor.query()
                while points.shape[0] != 0:
                    pointsf = torch.FloatTensor(points).to(self.device)
                    # Normalize to bounding box
                    pointsf = pointsf / mesh_extractor.resolution
                    pointsf = box_size * (pointsf - 0.5) + p
                    # print('################', mesh_extractor.resolution, pointsf)
                    values = self.eval_points(
                        pointsf, z, c, **kwargs).cpu().numpy()
                    values = values.astype(np.float64)
                    mesh_extractor.update(points, values)
                    points = mesh_extractor.query()
                value_grid = mesh_extractor.to_dense()
                # print('%%%%%%%%%%%%%%%%%', value_grid.shape, value_grid)
            vertices_p, faces_p, normal_p = self.extract_mesh2(p, value_grid, box_size, z, c)
            faces_p = faces_p + vertices_index
            vertices_index = len(vertices_p) + vertices_index
            vertices.append(vertices_p)
            faces.append(faces_p)
            if self.with_normals:
                normals.append(normal_p)

        vertices = np.concatenate(vertices, axis=0)
        faces = np.concatenate(faces, axis=0)

        mesh = trimesh.Trimesh(vertices, faces,
                               vertex_normals=normals,
                               process=False)
        return mesh

    def generate_from_latent(self, z, c=None,gt=False, stats_dict={}, **kwargs):
        ''' Generates mesh from latent.

        Args:
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        t0 = time.time()
        converge_steps = 0
        if converge_steps == 0:
            occ_psgn, points_psgn = self.model.decode2(z, c, **kwargs)
        else:
            points_psgn = self.model.decoder_psgn(z, c)
            for i in range(converge_steps):
                occ_psgn, points_psgn, normal = self.estimate_normals_oc(points_psgn, z, c)
                step= normal.permute(0,2,1) * torch.sign(occ_psgn.logits - threshold)
                points_psgn = points_psgn - 0.1 * step.permute(0,2,1)
                # print('%%%%%%%%%%%%%%%%%%%', occ_psgn.probs-self.threshold)

        mesh = self.partial_mesh(points_psgn, z, c, **kwargs)
        # mesh = 0
        #
        # points_psgn = points_psgn.squeeze(0).detach().cpu().numpy()
        # # occ_psgn = occ_psgn.squeeze(0).detach().cpu().numpy()
        # mesh = meshlab_poisson(points_psgn)
        # in_points = points_psgn[occ_psgn>self.threshold]
        # out_points = points_psgn[occ_psgn<self.threshold]
        # max_min_dist = self.max_min_dist(points_psgn)
        in_points = 0
        out_points = 0


        mesh2 = 0
        if gt == True:
            # Compute bounding box size
            box_size = 1 + self.padding
            # Shortcut
            if self.upsampling_steps == 0:
                nx = self.resolution0
                pointsf = box_size * make_3d_grid(
                    (-0.5,)*3, (0.5,)*3, (nx,)*3
                )
                values = self.eval_points(pointsf, z, c, **kwargs).cpu().numpy()
                value_grid = values.reshape(nx, nx, nx)
            else:
                mesh_extractor = MISE(
                    self.resolution0, self.upsampling_steps, threshold)

                points = mesh_extractor.query()

                while points.shape[0] != 0:
                    # Query points
                    pointsf = torch.FloatTensor(points).to(self.device)
                    # Normalize to bounding box
                    pointsf = pointsf / mesh_extractor.resolution
                    # print('$$$$$$$$$$$$$$$$', pointsf)
                    pointsf = box_size * (pointsf - 0.5)
                    # Evaluate model and update
                    values = self.eval_points(
                        pointsf, z, c, **kwargs).cpu().numpy()
                    values = values.astype(np.float64)
                    mesh_extractor.update(points, values)
                    points = mesh_extractor.query()

                value_grid = mesh_extractor.to_dense()

            # Extract mesh
            stats_dict['time (eval points)'] = time.time() - t0

            mesh2 = self.extract_mesh(value_grid, z, c, stats_dict=stats_dict)
        return mesh, mesh2, out_points, in_points

    def estimate_normals_oc(self, points, z, c=None):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        # device = self.device
        # z = z.unsqueeze(0)
        # occ_psgn, points = self.model.decode2(z, c)

        points = points.detach()
        points.requires_grad_()
        occ_psgn = self.model.decode_original(points, z, c)
        out = occ_psgn.logits.sum()
        ##################################
        out.backward()
        ##################################3
        # ni = points.grad.permute(0, 2, 1) * occ_psgn.logits
        # ni = ni.permute(0, 2, 1)
        ni = points.grad
        ni = ni / torch.norm(ni, dim=-1, keepdim=True)**2


        # occ_hat = torch.cat(occ_hats, dim=0).detach()
        # normals = torch.cat(normals, axis=0).permute(1,0).detach()
        return occ_psgn, points, ni

    def eval_points(self, p, z, c=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat, occ_psgn, points = self.model.decode(pi, z, c, **kwargs)

            occ_hats.append(occ_hat.logits.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat
    def extract_mesh2(self, point, occ_hat, box_size, z, c=None):
        n_x, n_y, n_z = occ_hat.shape
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        # occ_hat_padded = np.pad(
        #     occ_hat, 1, 'constant', constant_values=-1e6)
        occ_hat_padded = occ_hat
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        # vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
        vertices = box_size * (vertices - 0.5)

        point = point.detach().cpu().numpy()
        vertices = vertices + point

        if self.with_normals and not vertices.shape[0] == 0:
            normals = self.estimate_normals(vertices, z, c)

        else:
            normals = None

        return vertices, triangles, normals

    def extract_mesh(self, occ_hat, z, c=None, stats_dict=dict()):
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
            normals = self.estimate_normals(vertices, z, c)
            stats_dict['time (normals)'] = time.time() - t0

        else:
            normals = None

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
            self.refine_mesh(mesh, occ_hat, z, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh

    def estimate_normals(self, vertices, z, c=None):
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
        z, c = z.unsqueeze(0), c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decode(vi, z, c).logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, z, c=None):
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
                self.model.decode(face_point.unsqueeze(0), z, c).logits
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

FILTER_SCRIPT_RECONSTRUCTION = '''
<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Surface Reconstruction: Ball Pivoting">
  <Param value="0" type="RichAbsPerc" max="1.4129" name="BallRadius" description="Pivoting Ball radius (0 autoguess)" min="0" tooltip="The radius of the ball pivoting (rolling) over the set of points. Gaps that are larger than the ball radius will not be filled; similarly the small pits that are smaller than the ball radius will be filled."/>
  <Param value="20" type="RichFloat" name="Clustering" description="Clustering radius (% of ball radius)" tooltip="To avoid the creation of too small triangles, if a vertex is found too close to a previous one, it is clustered/merged with it."/>
  <Param value="90" type="RichFloat" name="CreaseThr" description="Angle Threshold (degrees)" tooltip="If we encounter a crease angle that is too large we should stop the ball rolling"/>
  <Param value="false" type="RichBool" name="DeleteFaces" description="Delete intial set of faces" tooltip="if true all the initial faces of the mesh are deleted and the whole surface is rebuilt from scratch, other wise the current faces are used as a starting point. Useful if you run multiple times the algorithm with an incrasing ball radius."/>
 </filter>
</FilterScript>
'''


def meshlab_poisson(pointcloud):
    r''' Runs the meshlab ball pivoting algorithm.

    Args:
        pointcloud (numpy tensor): input point cloud
    '''
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'script.mlx')
        input_path = os.path.join(tmpdir, 'input.ply')
        output_path = os.path.join(tmpdir, 'out.off')

        # Write script
        with open(script_path, 'w') as f:
            f.write(FILTER_SCRIPT_RECONSTRUCTION)

        # Write pointcloud
        export_pointcloud(pointcloud, input_path, as_text=False)

        # Export
        env = os.environ
        subprocess.Popen('meshlabserver -i ' + input_path + ' -o '
                         + output_path + ' -s ' + script_path,
                         env=env, cwd=os.getcwd(), shell=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         ).wait()
        mesh = trimesh.load(output_path, process=False)

    return mesh
