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

file_path = '/home/wenjing/Downloads/ShapeNet/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/model.binvox'
with open(file_path, 'rb') as f:
    voxels = binvox_rw.read_as_3d_array(f)
voxels = voxels.data.astype(np.float32)
# voxels[:,15,15] = 0
# print(voxels.shape, voxels)
# x = np.linspace(-0.5, 0.5,num=voxels.shape[0])
# y = np.linspace(-0.5, 0.5,num=voxels.shape[1])
# points_xy = np.concatenate([x,y], axis=0)
# print(points_xy.shape, points_xy)


points_xy = []
for i in np.linspace(-0.5, 0.5,num=voxels.shape[0]):
    for j in np.linspace(-0.5, 0.5,num=voxels.shape[1]):
        points_xy.append([i,j])

# points_xy = np.array(points_xy)
points_xy = np.array(points_xy).astype(np.float32)
# points_xy += 1e-4 * np.random.randn(*points_xy.shape)
# print(points_xy.shape)

voxels = voxels.transpose(2,1,0)
voxels = voxels.reshape(1024, 32)
# voxels_2 = voxels.reshape(1024,32)
# voxels_2 = voxels_2[0:300]
# voxels_2 = np.expand_dims(voxels_2, axis=0)

# print(voxels_2.shape)
#
# # print(voxels.shape)
x = np.array([[[1,2],[3,4]],[[5,6], [7,8]]])
print(x.shape)
x=x.transpose(0,1,2)
# x[:,1,1] = 0
# x = x.reshape(4,2)
print(x)


# Create plot
# fig = plt.figure()
# ax = fig.gca(projection=Axes3D.name)
# voxels = voxels.transpose(2, 0, 1)
# # voxels_2 = voxels_2.transpose(2, 0, 1)
# ax.voxels(voxels, edgecolor='k')
# # ax.voxels(voxels_2, edgecolor='r')
# ax.set_xlabel('Z')
# ax.set_ylabel('X')
# ax.set_zlabel('Y')
# # ax.view_init(elev=30, azim=45)
# plt.show()
