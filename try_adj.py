import pickle
import pandas as pd
import numpy as np
import os
import torch
import imageio
from PIL import Image
from torch.nn import functional as F
import matplotlib.pyplot as plt
# with open('/home/wenjing/Desktop/onet_2d_64/pretrained/time_generation.pkl', 'rb') as f:
#     data = pickle.load(f)
#
# print(data)
def sample_plane_feature(xy, c):
    # xy = p / 256  # normalize to the range of (0, 1)
    xy = xy[:, :, None].float()
    vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
    c = F.grid_sample(c, vgrid, align_corners=True).squeeze(-1)
    return c
# d = pd.read_pickle('/home/wenjing/Desktop/onet/pretrained/time_generation.pkl')
# d.to_csv('/home/wenjing/Desktop/onet/pretrained/time_generation.csv')
# W = 13
# H = 13
# i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
#                               torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
# i = i.t() #transpose
# j = j.t()
# print(i)
# print('##', j)
# i = i.reshape(-1, 1)
# j = j.reshape(-1, 1)
# print(')', i)
# print('!!!!!', j)
# filename = '/home/wenjing/Downloads/DVR/02933112/108295ef2f00c9aa9d886ab2fa5ee681/visual_hull_depth/00020001.exr'
filename2 = '/home/wenjing/Downloads/DVR/04379243/ccb1c5fecc863084391e4d6c585a697a/depth/00030001.exr'
filename_mask = '/home/wenjing/Downloads/DVR/04379243/ccb1c5fecc863084391e4d6c585a697a/mask/0003.png'
# depth1 = np.array(imageio.imread(filename)).astype(np.float32)
depth = np.array(imageio.imread(filename2)).astype(np.float32)
# print(depth.shape, depth)
depth = depth.reshape(depth.shape[0], depth.shape[1], -1)[:, :, 0]
# depth = torch.tensor(depth)
# visual_hull_depth = visual_hull_depth.reshape(visual_hull_depth.shape[0], visual_hull_depth.shape[1], -1)[:, :, 0]
mask = np.array(Image.open(filename_mask)).astype(np.bool)
# # print(mask)
# mask = mask.reshape(mask.shape[0], mask.shape[1], -1)[:, :, 0]
# mask = mask.astype(np.float32)
# mask = torch.tensor(mask).to('cuda').unsqueeze(0).unsqueeze(0)
# points_xy = torch.rand(1, 100, 2).to('cuda')
# occ_gt = sample_plane_feature(points_xy, mask).squeeze(1)
#
# occ_gt[occ_gt==1] = 0
# print(torch.max(occ_gt), occ_gt)
# depth = depth[mask]
# depth = depth[depth<0]
# print(depth)
print(np.max(depth), np.min(depth))
# visual_hull_depth = np.abs(depth - visual_hull_depth)
# im = Image.fromarray(visual_hull_depth)
# im.show()
im2 = Image.fromarray(depth)
im2.show()
