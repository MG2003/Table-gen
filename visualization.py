import numpy as np
from matplotlib import pyplot as plt

def visualize(indices):
    x_ax = [voxel[0] for voxel in indices]
    y_ax = [voxel[1] for voxel in indices]
    z_ax = [voxel[2] for voxel in indices]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim([0, 75])
    ax.set_ylim([0, 75])
    ax.set_zlim([0, 75])
    ax.scatter(x_ax, y_ax, z_ax)
    plt.show()