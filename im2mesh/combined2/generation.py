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
from im2mesh.utils.io import export_pointcloud
import tempfile
import subprocess
import os


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

    def __init__(self, model,model_ex, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1, sample=False,
                 simplify_nfaces=None,
                 preprocessor=None):
        self.model_in = model.to(device)
        self.model_ex = model_ex
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

    def generate_mesh(self, data, return_stats=True):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        ######psgn#####################################################33
        self.model_ex.eval()
        device = self.device

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)

        with torch.no_grad():
            points = self.model_ex(inputs).squeeze(0)

        # points = points.cpu().numpy()

        #################################################################3
        self.model_in.eval()
        # device = self.device
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
            c = self.model_in.encode_inputs(inputs)
        stats_dict['time (encode inputs)'] = time.time() - t0

        z = self.model_in.get_z_from_prior((1,), sample=self.sample).to(device)
        ###########################################################
        mesh = self.generate_from_latent2(z, c, points, stats_dict=stats_dict, **kwargs)
        ######################################################################

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh

    def generate_from_latent2(self, z, c=None, points=None, stats_dict={}, **kwargs):
        for i in range(200):
            normals, values = self.estimate_normals_oc(points, z, c)
            # print('**********', values[abs(values) > 1].shape)
            points = points - torch.mul(normals,values).permute(1,0)
        normals, values = self.estimate_normals_oc(points, z, c)

        points = points.cpu().numpy()
        mesh = meshlab_poisson(points)
        # mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces, process=False)
        return mesh

    def generate_from_latent(self, z, c=None, points=None, stats_dict={}, **kwargs):
        ''' Generates mesh from latent.

        Args:
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            #############################
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )

            ####################################
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

        mesh = self.extract_mesh(value_grid, z, c, stats_dict=stats_dict)
        return mesh

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
                occ_hat = self.model.decode(pi, z, c, **kwargs).logits

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat

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
        # print('!!!!!!', len(vertices))
        # print('%%%%%', triangles)
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

    def estimate_normals_oc(self, vertices, z, c=None):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        device = self.device
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        z = z.unsqueeze(0)
        occ_hats = []
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            # print('$$$$$$$$$4',vi.grad_fn)
            vi.requires_grad_()
            occ_hat = self.model_in.decode(vi, z, c).logits
            # occ_hats.append(occ_hat.squeeze(0).detach())
            occ_hats.append(occ_hat.squeeze(0))
            out = occ_hat.sum()
            ##################################
            out.backward()
            ##################################3
            ni = vi.grad
            # print('$$$$$$', vi.grad_fn)
            ni = ni / torch.norm(ni, dim=-1, keepdim=True) ** 2
            ni = ni.squeeze(0)
            normals.append(ni)

        occ_hat = torch.cat(occ_hats, dim=0).detach()
        normals = torch.cat(normals, axis=0).permute(1, 0).detach()
        return normals, occ_hat

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

