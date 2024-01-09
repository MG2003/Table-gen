import numpy as np
from matplotlib import pyplot as plt
import os
import open3d as o3d

NUM_MESHES = 500

OUT_PATH = "voxelized_models\\"

for i in range(NUM_MESHES):
    cube = "Cube" + str(i)
    path = os.getcwd() + '\\sdf_models\\' + cube + '\\meshes\\visual\\' + cube + '.obj'

    mesh = o3d.io.read_triangle_mesh(path)
    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
           center=mesh.get_center())
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                              voxel_size=0.015625)
    voxels = voxel_grid.get_voxels()
    indices = np.asarray([voxel.grid_index for voxel in voxels])

    """
    x_ax = [voxel[0] for voxel in indices]
    y_ax = [voxel[1] for voxel in indices]
    z_ax = [voxel[2] for voxel in indices]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x_ax, y_ax, z_ax)
    plt.show()
    """