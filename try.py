import numpy as np
import pickle

# base_mesh = np.loadtxt('im2mesh/pix2mesh/ellipsoid/face1.obj', dtype='|S32')
# print(base_mesh)

ellipsoid = pickle.load(open('im2mesh/pix2mesh/ellipsoid/info_ellipsoid.dat', 'rb'), encoding='latin1')
print(ellipsoid[1][1])
# ellipsoid[0] initial coordinates
# ellipsoid[1][1] ellipsoid[2][1] ellipsoid[3][1] support_array (tnsor): sparse weighted adjencency matrix
#                 with non-zero entries on the diagonal
# ellipsoid[4][0] ellipsoid[4][1] # IDs for the first/second unpooling operation,pool_idx_array (tensor): vertex IDs that should be combined to new
#             vertices