import os
import numpy as np
import torch
import imageio
from PIL import Image

cat_list = [
# '03001627',
# '02958343', #02958343/f9c1d7748c15499c6f2bd1c4e9adb41 empty directory
# '04256520',
# '02691156',
# '03636649', #lamp zero-size array to reduction operation maximum which has no identity
# '04401088',
# '04530566',
# '03691459',
# '02933112',
'04379243',
# '03211117',
# '02828884',
# '04090263',
]
for i in cat_list:
    print(i)
    type_path = '/home/wenjing/storage/DVR/ShapeNet/'+ i
    my_list = os.listdir(type_path)

    # # os.walk(type_path)
    # l = [x[0] for x in os.walk(type_path)]
    # print(l)
    depth_max = 0
    depth_min = 100
    for i in my_list:
        if 'lst' not in i:
            model_path = os.path.join(type_path, i)
            depth_path = os.path.join(model_path, 'depth')
            mask_path = os.path.join(model_path, 'mask')
            for i in range(10):
                depth_file = depth_path + '/000' + str(i) + '0001.exr'
                mask_file = mask_path + '/000' + str(i) + '.png'
                depth = np.array(imageio.imread(depth_file)).astype(np.float32)
                mask = np.array(Image.open(mask_file)).astype(np.bool)
                depth = depth[mask]
                d_max = np.max(depth)
                d_min = np.min(depth)
                if d_max > depth_max:
                    depth_max = d_max
                if d_min < depth_min:
                    depth_min = d_min
                if d_max > 2.4:
                    print('max', depth_file)
                if d_min < 0:
                    print('min', depth_file)
            for i in range(10, 24):
                depth_file = depth_path + '/00' + str(i) + '0001.exr'
                mask_file = mask_path + '/00' + str(i) + '.png'
                depth = np.array(imageio.imread(depth_file)).astype(np.float32)
                mask = np.array(Image.open(mask_file)).astype(np.bool)
                depth = depth[mask]
                d_max = np.max(depth)
                d_min = np.min(depth)
                if d_max > depth_max:
                    depth_max = d_max
                if d_min < depth_min:
                    depth_min = d_min

    print(depth_min, depth_max)

