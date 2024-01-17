import numpy as np
from matplotlib import pyplot as plt
import os
import open3d as o3d

NUM_MESHES = 500

HIGH_RES = 1/32
LOW_RES = 1/16

RESOLUTION = HIGH_RES
AXIS = 1/RESOLUTION - 1

OUT_PATH = "voxelized_arrays\\"

for i in range(NUM_MESHES):
    cube = "Cube" + str(i)
    path = os.getcwd() + '\\sdf_models\\' + cube + '\\meshes\\visual\\' + cube + '.obj'
    out = OUT_PATH + cube + ".npy"

    mesh = o3d.io.read_triangle_mesh(path)
    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
           center=mesh.get_center())
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                              voxel_size=RESOLUTION)
    voxels = voxel_grid.get_voxels()
    indices = np.asarray([voxel.grid_index for voxel in voxels])
    np.save(out, indices)
    



    
    x_ax = [voxel[0] for voxel in indices]
    y_ax = [voxel[1] for voxel in indices]
    z_ax = [voxel[2] for voxel in indices]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim([0, AXIS])
    ax.set_ylim([0, AXIS])
    ax.set_zlim([0, AXIS])
    ax.scatter(x_ax, y_ax, z_ax)
    plt.show()
    