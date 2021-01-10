import argparse
import trimesh
import numpy as np
import os
import glob
import sys
from multiprocessing import Pool
from functools import partial
# TODO: do this better
sys.path.append('..')
from im2mesh.utils import binvox_rw, voxels
from im2mesh.utils.libmesh import check_mesh_contains
import torch

parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('in_folder', type=str,
                    help='Path to input watertight meshes.')
parser.add_argument('--n_proc', type=int, default=0,
                    help='Number of processes to use.')

parser.add_argument('--resize', action='store_true',
                    help='When active, resizes the mesh to bounding box.')

parser.add_argument('--rotate_xz', type=float, default=0.,
                    help='Angle to rotate around y axis.')

parser.add_argument('--bbox_padding', type=float, default=0.,
                    help='Padding for bounding box')
parser.add_argument('--bbox_in_folder', type=str,
                    help='Path to other input folder to extract'
                         'bounding boxes.')

parser.add_argument('--pointcloud_folder', type=str,
                    help='Output path for point cloud.')
parser.add_argument('--pointcloud_size', type=int, default=100000,
                    help='Size of point cloud.')

parser.add_argument('--voxels_folder', type=str,
                    help='Output path for voxelization.')
parser.add_argument('--voxels_res', type=int, default=32,
                    help='Resolution for voxelization.')

parser.add_argument('--points_folder', type=str,
                    help='Output path for points.')
parser.add_argument('--points_size', type=int, default=100000,
                    help='Size of points.')
parser.add_argument('--points_uniform_ratio', type=float, default=1.,
                    help='Ratio of points to sample uniformly'
                         'in bounding box.')
parser.add_argument('--points_sigma', type=float, default=0.01,
                    help='Standard deviation of gaussian noise added to points'
                         'samples on the surfaces.')
parser.add_argument('--points_padding', type=float, default=0.1,
                    help='Additional padding applied to the uniformly'
                         'sampled points on both sides (in total).')

parser.add_argument('--mesh_folder', type=str,
                    help='Output path for mesh.')

parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite output.')
parser.add_argument('--float16', action='store_true',
                    help='Whether to use half precision.')
parser.add_argument('--packbits', action='store_true',
                help='Whether to save truth values as bit array.')
    
def main(args):
    input_files = glob.glob(os.path.join(args.in_folder, '*.off'))
    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            # print('$$$$$$$$$$$$$44', input_files)
            p.map(partial(process_path, args=args), input_files)
    else:
        for p in input_files:
            process_path(p, args)


def process_path(in_path, args):
    in_file = os.path.basename(in_path)
    modelname = os.path.splitext(in_file)[0]
    mesh = trimesh.load(in_path, process=False)

    # Determine bounding box
    if not args.resize:
        # Standard bounding boux
        loc = np.zeros(3)
        scale = 1.
    else:
        if args.bbox_in_folder is not None:
            in_path_tmp = os.path.join(args.bbox_in_folder, modelname + '.off')
            mesh_tmp = trimesh.load(in_path_tmp, process=False)
            bbox = mesh_tmp.bounding_box.bounds
        else:
            bbox = mesh.bounding_box.bounds

        # Compute location and scale
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 - args.bbox_padding)

        # Transform input mesh
        mesh.apply_translation(-loc)
        mesh.apply_scale(1 / scale)

        if args.rotate_xz != 0:
            angle = args.rotate_xz / 180 * np.pi
            R = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
            mesh.apply_transform(R)

    # Expert various modalities
    if args.pointcloud_folder is not None:
        export_pointcloud(mesh, modelname, loc, scale, args)

    if args.voxels_folder is not None:
        export_voxels(mesh, modelname, loc, scale, args)

    if args.points_folder is not None:
        export_points(mesh, modelname, loc, scale, args)

    if args.mesh_folder is not None:
        export_mesh(mesh, modelname, loc, scale, args)


def export_pointcloud(mesh, modelname, loc, scale, args):
    filename = os.path.join(args.pointcloud_folder,
                            modelname + '.npz')
    if not args.overwrite and os.path.exists(filename):
        print('Pointcloud already exist: %s' % filename)
        return

    points, face_idx = mesh.sample(args.pointcloud_size, return_index=True)
    normals = mesh.face_normals[face_idx]

    # Compress
    if args.float16:
        dtype = np.float16
    else:
        dtype = np.float32

    points = points.astype(dtype)
    normals = normals.astype(dtype)

    print('Writing pointcloud: %s' % filename)
    np.savez(filename, points=points, normals=normals, loc=loc, scale=scale)


def export_voxels(mesh, modelname, loc, scale, args):
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot create voxelization.' % modelname)
        return

    filename = os.path.join(args.voxels_folder, modelname + '.binvox')

    if not args.overwrite and os.path.exists(filename):
        print('Voxels already exist: %s' % filename)
        return

    res = args.voxels_res
    voxels_occ = voxels.voxelize(mesh, res)

    voxels_out = binvox_rw.Voxels(voxels_occ, (res,) * 3,
                                  translate=loc, scale=scale,
                                  axis_order='xyz')
    print('Writing voxels: %s' % filename)
    with open(filename, 'bw') as f:
        voxels_out.write(f)

def make_2d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])

    pxs = pxs.view(-1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1).expand(*shape).contiguous().view(size)

    p = torch.stack([pxs, pys], dim=1)

    return p

def export_points(mesh, modelname, loc, scale, args):
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot sample points.' % modelname)
        return

    filename = os.path.join(args.points_folder, modelname + '.npz')

    if not args.overwrite and os.path.exists(filename):
        print('Points already exist: %s' % filename)
        return

    n_points_uniform = int(args.points_size * args.points_uniform_ratio)
    n_points_surface = args.points_size - n_points_uniform

    boxsize = 1 + args.points_padding
    ##################################################################
    # points_uniform = np.random.rand(n_points_uniform, 3)
    # 2d onet
    # num_xy = 2500
    # num_z = 128
    # points_xy = np.random.rand(num_xy, 2)
    # points_uniform_xy = np.repeat(points_xy, num_z, axis=0)
    # points_uniform_z = np.linspace(0, 1, num=num_z, endpoint=False)
    # points_uniform_z = np.expand_dims(points_uniform_z, axis=1)
    # points_uniform_z = np.repeat(points_uniform_z, num_xy, axis=1).transpose(1,0)
    # points_uniform_z = np.reshape(points_uniform_z, (-1,1))
    # points_uniform = np.concatenate([points_uniform_xy, points_uniform_z], axis=1)
    #1d onet
    num_x = 100
    num_yz = 65
    points_x_o = np.random.rand(num_x, 1)
    points_x = np.repeat(points_x_o, num_yz ** 2)
    points_x = np.expand_dims(points_x, axis=1)
    points_yz = make_2d_grid((0,)*2, (1,)*2, (num_yz,)*2).numpy()
    points_yz = np.repeat(points_yz, num_x, axis=0).reshape(num_x * num_yz ** 2, 2)
    points_uniform = np.concatenate([points_x, points_yz], axis=1)
    #################################################################
    points_uniform = boxsize * (points_uniform - 0.5)
    points_surface = mesh.sample(n_points_surface)
    points_surface += args.points_sigma * np.random.randn(n_points_surface, 3)
    points = np.concatenate([points_uniform, points_surface], axis=0)
    # print('$$$$$$$$', points.shape)
    occupancies = check_mesh_contains(mesh, points)

    # Compress
    if args.float16:
        dtype = np.float16
    else:
        dtype = np.float32

    # points = points.astype(dtype)
    #2d onet
    # points_xy = boxsize * (points_xy - 0.5)
    # points_xy = points_xy.astype(dtype)
    #1d onet
    points_x_o = boxsize * (points_x_o - 0.5)
    points_x_o = points_x_o.astype(dtype)
    if args.packbits:
        occupancies = np.packbits(occupancies)

    print('Writing points: %s' % filename)
    # np.savez(filename, points_xy=points_xy, occupancies=occupancies,
    #          loc=loc, scale=scale)
    np.savez(filename, points_x=points_x_o, occupancies=occupancies,
             loc=loc, scale=scale)


def export_mesh(mesh, modelname, loc, scale, args):
    filename = os.path.join(args.mesh_folder, modelname + '.off')    
    if not args.overwrite and os.path.exists(filename):
        print('Mesh already exist: %s' % filename)
        return
    print('Writing mesh: %s' % filename)
    mesh.export(filename)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
