from plyfile import PlyData, PlyElement
import numpy as np
import im2mesh.common as common
import torch
import trimesh

def write_ply(save_path, points, text=True):
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)

def fix_Rt_camera(Rt, loc, scale):
    ''' Fixes Rt camera matrix.

    Args:
        Rt (tensor): Rt camera matrix
        loc (tensor): location
        scale (float): scale
    '''
    batch_size = Rt.size(0)
    R = Rt[:, :, :3]
    t = Rt[:, :, 3:]

    scale = scale
    R_new = R * scale
    t_new = t + R @ loc.unsqueeze(2)

    Rt_new = torch.cat([R_new, t_new], dim=2)

    assert (Rt_new.size() == (batch_size, 3, 4))
    return Rt_new
#
# def fix_K_camera(K, img_size=137):
#     """Fix camera projection matrix.
#
#     This changes a camera projection matrix that maps to
#     [0, img_size] x [0, img_size] to one that maps to [-1, 1] x [-1, 1].
#
#     Args:
#         K (np.ndarray):     Camera projection matrix.
#         img_size (float):   Size of image plane K projects to.
#     """
#     # Unscale and recenter
#     scale_mat = torch.tensor([
#         [2./img_size, 0, -1],
#         [0, 2./img_size, -1],
#         [0, 0, 1.],
#     ], device=K.device, dtype=K.dtype)
#     K_new = scale_mat.view(1, 3, 3) @ K
#     return K_new
# data = np.load('/home/wenjing/Downloads/ShapeNet2/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/img_choy2016/cameras.npz')
# mesh_file = '/home/wenjing/Desktop/1a04e3eab45ca15dd86060f189eb133_4water.off'
# mesh = trimesh.load(mesh_file, process=False)

# data = np.load('/home/wenjing/storage/raw/ShapeNet.build/03001627/4_points/fffda9f09223a21118ff2740a556cc3.npz')
# data = np.load('/home/wenjing/Downloads/ShapeNet/02828884/78585e161ee3e39acb2a965e75be701c/points.npz')
data = np.load('/home/wenjing/Downloads/ShapeNet/03001627/97df0e7773e51feb331fc18393f04d2a/pointcloud.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item], data[item].shape)
# scale = data['scale']
# vertices = mesh.vertices * scale
# mesh_scaled = trimesh.Trimesh(vertices, mesh.faces,
#                                process=False)
# mesh_scaled.export('/home/wenjing/Desktop/scaled.off')
points = torch.tensor(data['points'].astype(np.float32)).unsqueeze(0)
# batch_size, D, T = points.size()
# z_coor = torch.linspace(0, 1, 65)[:-1].unsqueeze(1)
# z_coor = 1.1 * (z_coor - 0.5)
# z_coor = z_coor.expand(batch_size, D, *z_coor.size())
# xy_coor = points.unsqueeze(2)
# xy_coor = xy_coor.expand(-1, -1, 64, -1)
# points = torch.cat([xy_coor,z_coor], dim=3)
# points = points.view(batch_size, -1, 3)
# occupancies = np.unpackbits(data['occupancies'])
# # print(np.unique(occupancies), occupancies.shape)
# points = points.reshape(2500, 128, 3)
# print(points)
# occupancies = occupancies.reshape(2500, 128)
#
# n = 32
# interval=int(128/n)
# index = [i*interval for i in range(32)]
# print(index)
# occupancies = occupancies[:, index]
# print(occupancies.shape)


# n = np.array([[1,2,3],[4,5,6]])
# a = n[:,[1,2]]
# print(n.shape, a)
# x = []
# for i in range(len(points)):
#     x.append(points[i][1])
# x=np.array(x)
# print(x.shape, np.amax(x))
loc = torch.tensor(data['loc'].astype(np.float32)).unsqueeze(0)
scale = torch.tensor(data['scale'].astype(np.float32))
# # mat = np.load('/home/wenjing/Downloads/ShapeNet/02691156/1a04e3eab45ca15dd86060f189eb133/img_choy2016/cameras.npz')
mat = np.load('/home/wenjing/Downloads/ShapeNet/03001627/97df0e7773e51feb331fc18393f04d2a/img_choy2016/cameras.npz')
world_mat = torch.tensor(mat['world_mat_2'].astype(np.float32)).unsqueeze(0)
camera_mat = torch.tensor(mat['camera_mat_2'].astype(np.float32)).unsqueeze(0)
print(camera_mat)
print(world_mat)
# world_mat = fix_Rt_camera(world_mat, loc, scale)
points_transformed = common.transform_points(points, world_mat)
# points_projection = common.project_to_camera(points_transformed, camera_mat)
# dir = '/home/wenjing/Desktop/points.ply'
# print(points.shape)
# write_ply(dir, points[0])

# points_transformed = common.transform_points(points, world_mat)
#
# dir2 = '/home/wenjing/Desktop/pc_t2.ply'
# write_ply(dir2, points_transformed[0])

# camera_mat = torch.tensor(mat['camera_mat_0'].astype(np.float32)).unsqueeze(0)
# camera_mat = fix_K_camera(camera_mat)
# print(camera_mat)

