import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from im2mesh.common import (
    compute_iou, make_2d_grid
)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer
import im2mesh.common as common

class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5, eval_sample=False, z_resolution=32, camera=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.z_resolution = z_resolution
        self.camera = camera

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Compute elbo
        points_xy = data.get('points').to(device)
        occ_z = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points_xy.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)

        if self.camera:
            camera_args = common.get_camera_args(
                data, 'points.loc', 'points.scale', device=self.device)
            # # Transform GT data into camera coordinate system
            world_mat, camera_mat = camera_args['Rt'], camera_args['K']
            camera_position = world_mat[:, :3, -1]
        else:
            camera_position = None

        kwargs = {}
        # print('@@@@@@@@@@22', points.shape) (10, 2048, 3)

        with torch.no_grad():
            elbo, rec_error, kl = self.model.compute_elbo(
                points_xy, occ_z, inputs,camera_position=camera_position, **kwargs)

        eval_dict['loss'] = -elbo.mean().item()
        eval_dict['rec_error'] = rec_error.mean().item()
        eval_dict['kl'] = kl.mean().item()

        # Compute iou
        batch_size = points_xy.size(0)

        # print('@@@@@@@@@@', points_iou.shape) #(10, 100000, 3)

        with torch.no_grad():
            p_out = self.model(points_iou, inputs,
                               sample=self.eval_sample, camera_position=camera_position, **kwargs)


        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        # print('$$$$$$$$$$$$$', occ_iou_np.shape)
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou


        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_2d_grid(
                (-0.5 + 1/64,) * 2, (0.5 - 1/64,) * 2, (32,) * 2)
            # points_voxels = make_2d_grid(
            #     (-0.5,) * 2, (0.5,) * 2, (32,) * 2)
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)

            # print('$$$$$$$$$44', points_voxels.shape)  #(10, 32768, 3)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs,
                                   sample=self.eval_sample, camera_position=camera_position, **kwargs)


            voxels_occ = voxels_occ.permute(0, 3, 2, 1)
            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()
            # print('$$$$$$$$$$$$$$$$$', iou_voxels)

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        z_resolution = self.z_resolution

        batch_size = data['points'].size(0)
        inputs = data.get('inputs', torch.empty(batch_size, 0)).to(device)

        shape = (32, 32)
        p = make_2d_grid([-0.5] * 2, [0.5] * 2, shape).to(device)
        p = p.expand(batch_size, *p.size())
        # print('########', p.shape)  [12, 1024, 2]

        kwargs = {}

        # print('#########3', p.shape)  #(12, 32^2, 2)
        if self.camera:
            camera_args = common.get_camera_args(
                data, 'points.loc', 'points.scale', device=self.device)
            # # Transform GT data into camera coordinate system
            world_mat, camera_mat = camera_args['Rt'], camera_args['K']
            camera_position = world_mat[:, :3, -1]
        else:
            camera_position = None
        with torch.no_grad():
            p_r = self.model(p, inputs, sample=self.eval_sample, camera_position=camera_position, **kwargs)

        # print('$$$$$$$$$$$$$$', p_r.probs.shape) [12, 32, 1024]
        occ_hat = p_r.probs.view(batch_size, *shape, z_resolution)
        voxels_out = (occ_hat >= self.threshold).cpu().numpy()

        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(
                inputs[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(
                voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p_xy = data.get('points').to(device)
        occ_z = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p_xy.size(0), 0)).to(device)

        kwargs = {}
        # print('%%%%%%%%%%%%%%', inputs.shape) [64, 3, 224, 224]

        c = self.model.encode_inputs(inputs)
        # print('#############', c.shape) #[64, 256]
        if self.camera:
            camera_args = common.get_camera_args(
                data, 'points.loc', 'points.scale', device=self.device)
            # # Transform GT data into camera coordinate system
            world_mat, camera_mat = camera_args['Rt'], camera_args['K']
            camera_position = world_mat[:, :3, -1]
            c = torch.cat([c, camera_position], dim=1)

        q_z = self.model.infer_z(p_xy, occ_z, c, **kwargs)
        z = q_z.rsample()

        # KL-divergence
        kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        loss = kl.mean()


        # print('########3333', p_xy.shape)  [64, 2048, 2]

        logits = self.model.decode(p_xy, z, c, **kwargs).logits

        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ_z, reduction='none')
        loss = loss + loss_i.sum(-1).mean()

        return loss
