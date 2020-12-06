try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# # Extensions
# # pykdtree (kd tree)
# pykdtree = Extension(
#     'im2mesh.utils.libkdtree.pykdtree.kdtree',
#     sources=[
#         'im2mesh/utils/libkdtree/pykdtree/kdtree.c',
#         'im2mesh/utils/libkdtree/pykdtree/_kdtree_core.c'
#     ],
#     language='c',
#     extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
#     extra_link_args=['-lgomp'],
# )

# mcubes (marching cubes algorithm)
msquares_module = Extension(
    'im2mesh.utils.libmcubes.mcubes',
    sources=[
        'im2mesh/utils/libmsquares/msquares.pyx',
        'im2mesh/utils/libmsquares/pywrapper.cpp',
        'im2mesh/utils/libmsquares/marchingsquares.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# # triangle hash (efficient mesh intersection)
# triangle_hash_module = Extension(
#     'im2mesh.utils.libmesh.triangle_hash',
#     sources=[
#         'im2mesh/utils/libmesh/triangle_hash.pyx'
#     ],
#     libraries=['m']  # Unix-like specific
# )

# mise (efficient mesh extraction)
mise_2d_module = Extension(
    'im2mesh.utils.libmise.mise',
    sources=[
        'im2mesh/utils/libmise_2d/mise_2d.pyx'
    ],
)

# # simplify (efficient mesh simplification)
# simplify_mesh_module = Extension(
#     'im2mesh.utils.libsimplify.simplify_mesh',
#     sources=[
#         'im2mesh/utils/libsimplify/simplify_mesh.pyx'
#     ]
# )

# # voxelization (efficient mesh voxelization)
# voxelize_module = Extension(
#     'im2mesh.utils.libvoxelize.voxelize',
#     sources=[
#         'im2mesh/utils/libvoxelize/voxelize.pyx'
#     ],
#     libraries=['m']  # Unix-like specific
# )

# # DMC extensions
# dmc_pred2mesh_module = CppExtension(
#     'im2mesh.dmc.ops.cpp_modules.pred2mesh',
#     sources=[
#         'im2mesh/dmc/ops/cpp_modules/pred_to_mesh_.cpp',
#     ]
# )

# dmc_cuda_module = CUDAExtension(
#     'im2mesh.dmc.ops._cuda_ext',
#     sources=[
#         'im2mesh/dmc/ops/src/extension.cpp',
#         'im2mesh/dmc/ops/src/curvature_constraint_kernel.cu',
#         'im2mesh/dmc/ops/src/grid_pooling_kernel.cu',
#         'im2mesh/dmc/ops/src/occupancy_to_topology_kernel.cu',
#         'im2mesh/dmc/ops/src/occupancy_connectivity_kernel.cu',
#         'im2mesh/dmc/ops/src/point_triangle_distance_kernel.cu',
#     ]
# )

# Gather all extension modules
ext_modules = [
    msquares_module,
    mise_2d_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    },
    include_dirs=[numpy.get_include()]
)
