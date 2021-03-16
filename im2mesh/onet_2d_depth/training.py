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
import im2mesh.common2 as common2
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
                 vis_dir=None, threshold=0.5, eval_sample=False, z_resolution=32, camera=False,
                 n_training_points=256, n_eval_points=1000):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.z_resolution = z_resolution
        self.camera = camera

        self.n_training_points = n_training_points
        self.n_eval_points = n_eval_points

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
        eval_dict = {}

        with torch.no_grad():
            eval_dict = self.compute_loss(
                data, eval_mode=True)

        for (k, v) in eval_dict.items():
            eval_dict[k] = v.item()


        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        z_resolution = self.z_resolution

        (img, mask_img, depth_img, world_mat, camera_mat, scale_mat,
         inputs) = self.process_data_dict(data)
        batch_size, _, H, W = img.size()

        # get camera origin in world coordinate
        camera_ori_world = common2.origin_to_world(
            1024, camera_mat, world_mat, scale_mat)

        shape = (32, 32)
        points_xy = make_2d_grid([-1.0] * 2, [1.0] * 2, shape).to(device)
        points_xy = points_xy.expand(batch_size, *points_xy.size())

        focal = camera_mat[0, 0, 0]
        ray_dir = common2.get_rays(points_xy, focal, device=device)

        kwargs = {}

        with torch.no_grad():
            p_r = self.model(camera_ori_world, points_xy, ray_dir, img)

        occ_hat = p_r.probs.view(batch_size, *shape, z_resolution)
        voxels_out = (occ_hat >= self.threshold).cpu().numpy()

        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(
                img[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(
                voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))

    def process_data_dict(self, data):
        ''' Processes the data dictionary and returns respective tensors

        Args:
            data (dictionary): data dictionary
        '''
        device = self.device

        # Get "ordinary" data
        img = data.get('img').to(device)
        mask_img = data.get('img.mask').unsqueeze(1).to(device)
        world_mat = data.get('img.world_mat').to(device)
        camera_mat = data.get('img.camera_mat').to(device)
        scale_mat = data.get('img.scale_mat').to(device)
        depth_img = data.get('img.depth', torch.empty(1, 0)
                             ).unsqueeze(1).to(device)
        inputs = data.get('inputs', torch.empty(1, 0)).to(device)

        # points_xy = data.get('img.points_xy').to(device)

        return (img, mask_img, depth_img, world_mat, camera_mat, scale_mat,
                inputs)
    def sample_plane_feature(self, xy, c):
        # xy = p / 256  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, align_corners=True).squeeze(-1)
        return c

    def compute_loss(self, data, eval_mode=False):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        n_points = self.n_eval_points if eval_mode else self.n_training_points
        # n_points = self.n_training_points
        # h = 12
        # w = 12
        # n_points = h * w
        depth_range = [0, 2.4]
        num_samples = self.z_resolution
        depth_array = torch.linspace(depth_range[0], depth_range[1], num_samples)
        # empty_depth = torch.tensor([10.])
        # depth_array = torch.cat([depth_array, empty_depth], dim=0)
        depth_array = depth_array.to(device)
        depth_array_2 = torch.linspace(depth_range[0], depth_range[1], num_samples)
        depth_array_2 = depth_array_2.to(device)
        # Process data dictionary
        (img, mask_img, depth_img, world_mat, camera_mat, scale_mat,
         inputs) = self.process_data_dict(data)
        depth_img[mask_img==0] = 100.


        batch_size, _, H, W = img.size()

        points_xy = torch.rand(batch_size, n_points, 2).to(device)

        # get camera origin in world coordinate
        camera_ori_world = common2.origin_to_world(
            n_points, camera_mat, world_mat, scale_mat)

        # get image pixel/ray direction in camera viewer coordinate
        focal = camera_mat[0, 0, 0]
        # print('))))))', focal)
        ray_dir = common2.get_rays(points_xy, focal, device=device)
        # points_xy = points_xy.expand(batch_size, -1, -1)

        # ground truth depth (B, n_points)
        depth_gt = self.sample_plane_feature(points_xy, depth_img).squeeze(1)
        #ground truth occupancy
        occ_gt = self.sample_plane_feature(points_xy, mask_img).squeeze(1)
        # print(oc_gt.shape, oc_gt)
        # depth_gt_finite = depth_gt[occ_gt==1]


        # obtain occupancies along the ray
        c, c_local = self.model.encode_inputs(img)

        probs = self.model.decode(camera_ori_world, points_xy, ray_dir, c, c_local).probs # (B, n_points, num_samples)

        ## depth loss
        probs = probs.unsqueeze(2)
        empty_probs = 1 - probs
        acc_empty_probs = [probs[:, :, :, 0]]
        for i in range(1, num_samples):
            acc_empty_prob = torch.ones(batch_size, n_points, 1).to(device)
            for j in range(i):
                acc_empty_prob = acc_empty_prob * empty_probs[:, :, :, j]
            acc_empty_prob = acc_empty_prob * probs[:, :, :, i]
            acc_empty_probs.append(acc_empty_prob)

        all_empty_prob = torch.ones(batch_size, n_points, 1).to(device)
        for i in range(num_samples):
            all_empty_prob = all_empty_prob * empty_probs[:, :, :, i]

        occ_prob = 1 - all_empty_prob.squeeze(2)
        acc_empty_probs = torch.cat(acc_empty_probs, dim=2)
        # depth_gt (B, n_points)
        # depth_array (num_samples)
        depth_gt = depth_gt.unsqueeze(2).expand(-1, -1, num_samples)
        depth_array = depth_array.unsqueeze(0).unsqueeze(0).expand(batch_size, n_points, -1)
        # depth_array_2 = depth_array_2.unsqueeze(0).unsqueeze(0).expand(batch_size, n_points, -1)
        loss = {}
        # # mask_inf = torch.isinf(depth_gt)
        # mask = (depth_gt==100)
        # depth_gt[mask] = 0
        # print(torch.max(depth_gt))
        # depth_array_2[depth_gt!=100] = 0
        # depth_gt = depth_gt + depth_array_2
        # # depth_gt  = depth_gt.detach()
        depth_loss = acc_empty_probs * torch.abs((depth_gt - depth_array))
        depth_loss = depth_loss.sum(-1)
        depth_loss = depth_loss[occ_gt==1]
        # depth_loss = depth_loss[occ_gt!=0]
        depth_loss = depth_loss.mean()

        # occupancy loss
        occupancy_loss = F.binary_cross_entropy(
            occ_prob, occ_gt, reduction='none').mean()
        # occupancy_loss = 0


        loss['loss'] = depth_loss + occupancy_loss
        return loss if eval_mode else loss['loss']
