import torch
import os
import numpy as np
from torch.utils.data import Dataset

class VoxelDataset(Dataset):
    def __init__(self, path, args):
        self.path = path
        self.directory = os.listdir(self.path)
        self.args = args

    def __len__(self):
        return len(self.directory)

    def __getitem__(self, idx):
        voxels = np.load(self.path + self.directory[idx])
        return torch.FloatTensor(voxels)