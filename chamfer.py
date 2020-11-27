import torch
from im2mesh.utils.libkdtree import KDTree
import numpy as np

def get_nearest_neighbors_indices_batch(points_src, points_tgt, k=1):
    ''' Returns the nearest neighbors for point sets batchwise.

    Args:
        points_src (numpy array): source points
        points_tgt (numpy array): target points
        k (int): number of nearest neighbors to return
    '''

    kdtree = KDTree(p2)
    dist, idx = kdtree.query(p1, k=1)

    return idx, dist


# p1 = np.array([[2,1],[1,4],[3,5]])

p2 = np.array([[28,54,98],[5,2,2],[24,54,98], [43,54,99]])
# p2=np.array([[28,5,34,34],[45,2,43,4],[43,1,43,54]])
for i, p1 in enumerate(p2):
    print(p1)
    p3 = np.delete(p2, i, 0)
    print('########', p3)
    p1 = np.expand_dims(p1, axis=0)
    indices, dist = get_nearest_neighbors_indices_batch(p1, p3)
    print('@@@', dist)


#2 1 3
#1 4 5
#5 2 1