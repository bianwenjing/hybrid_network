import torch
from im2mesh.utils.libkdtree import KDTree
import numpy as np
import logging
from copy import deepcopy


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

def get_rays(points_xy, focal, device=None):
    points_xy = points_xy - 0.5 # change xy from range (0, 1) to (-0.5, 0.5)
    i = points_xy[:, :, 0]
    j = points_xy[:, :, 1]
    # dirs = torch.stack([i / focal, -j / focal, -torch.ones_like(i)], -1)

    dirs_x = i / focal  # (H, W)
    dirs_y = -j / focal  # (H, W)
    # dirs_z = torch.ones(H, W, dtype=torch.float32)  # (H, W)
    ray_dir_cam = torch.stack([dirs_x, dirs_y, torch.ones_like(dirs_x)], dim=-1)  # (H, W, 3)

    return ray_dir_cam