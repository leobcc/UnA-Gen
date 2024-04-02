import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter
import trimesh

# Mesh handling functions

def export_mesh(vertices):
    '''Export mesh from vertices to .obj file.'''
    mesh = trimesh.Trimesh(vertices)
    mesh.export('mesh.obj')

# Matrix mapping handling in sparse notation

def initialize_matrix_mapping(mmap_dim, mmap_res, device):
    matrix_mapping = np.ones((mmap_dim, mmap_dim, mmap_dim))
    matrix_mapping = torch.from_numpy(matrix_mapping).to(device)
    #indices = torch.nonzero(tensor)

    return matrix_mapping

# Unet parts

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
       x1 = self.up(x1)
       x = torch.cat([x1, x2], 1)
       return self.conv(x)