import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import DoubleConv, DownSample, UpSample, PixelShuffle3d, PixelUnshuffle3d
from lib.deformer import SMPLDeformer, skinning
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from torchvision.utils import make_grid, save_image
from torchvision import transforms, models
import time
from lib.utils import get_camera_loc, coordinates_projection_map, get_cam, get_rays, get_uv, upsample_matrix, quat_to_rot, axis_angle_to_rotation_matrix, get_global_transformation
from lib.utils import stable_softmax, print_grad, world_to_pixel_batch, undistort_points
from collections import OrderedDict
import wandb
#import segmentation_models_pytorch as smp
from scipy.ndimage import binary_dilation, binary_erosion
from ray_caster import RayCaster
import yaml
import trimesh
from skimage import measure
import random
import pyvista as pv

import cv2
from torchvision.transforms import Compose
import sys
sys.path.append('/home/lbocchi/UnA-Gen/supp_repos/Depth_Anything_main/')
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-4])  # Remove the last layer (classification layer)
        
    def forward(self, x):
        x = self.resnet(x)
        return x

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 4)
        self.down_convolution_2 = DownSample(4, 8)
        self.down_convolution_3 = DownSample(8, 16)
        #self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(16, 32)

        #self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(32, 16)
        self.up_convolution_3 = UpSample(16, 8)
        self.up_convolution_4 = UpSample(8, 4)
        
        self.out = nn.Conv2d(in_channels=4, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        #down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p3)

        #up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(b, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out
    
class CNNEncoder(nn.Module):
    def __init__(self, input_channels=3, output_channels=32):
        super(CNNEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # First convolutional layer (Input: [batch_size, input_channels, H, W])
            nn.Conv2d(input_channels, 4, kernel_size=3, stride=1, padding=1),  # Output: [batch_size, 64, H, W]
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            
            # Second convolutional layer (no change in spatial size)
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),  # Output: [batch_size, 128, H, W]
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            # Third convolutional layer (no change in spatial size)
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # Output: [batch_size, 128, H, W]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Fourth convolutional layer (no change in spatial size)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Output: [batch_size, 256, H, W]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Fifth convolutional layer (no change in spatial size)
            nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1),  # Output: [batch_size, output_channels, H, W]
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.encoder(x)
        return x

class OccupancyField_cnn(nn.Module):
    def __init__(self, in_features=128, out_channels=1):
        super(OccupancyField_cnn, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//2, out_channels=in_features//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//2, out_channels=in_features//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//2, out_channels=in_features//2, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//2, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),     
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class RGBfield_cnn(nn.Module):
    def __init__(self, in_features=128, out_channels=3):
        super(RGBfield_cnn, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//2, out_channels=in_features//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//2, out_channels=in_features//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//2, out_channels=in_features//2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//2, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_r = nn.Sequential(
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),           
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),       
        )
        self.conv_b = nn.Sequential(
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),         
        )

    def forward(self, x):
        x = self.conv(x)
        x_r = self.conv_r(x)
        x_g = self.conv_g(x)
        x_b = self.conv_b(x)
        x = torch.cat((x_r, x_g, x_b), dim=1)

        return x

class OccupancyField_cnn_v1(nn.Module):
    def __init__(self, in_features=128, hidden_features=32, out_channels=1):
        super(OccupancyField_cnn_v1, self).__init__()

        self.c_conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=hidden_features, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),     
        )        
        input_dim = in_features-4+hidden_features
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=(input_dim)//4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),     
        )

    def forward(self, x):
        x_c = x[:,:4]
        x_c = self.c_conv(x_c)

        x = torch.cat((x_c, x[:,4:]), dim=1)
        x = self.conv(x)

        return x
    
class RGBfield_cnn_v1(nn.Module):
    def __init__(self, in_features=128, hidden_features=32, out_channels=3):
        super(RGBfield_cnn_v1, self).__init__()

        self.c_conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=hidden_features, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),     
        )
        input_dim = in_features-4+hidden_features
        self.conv_r = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=(input_dim)//4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),               
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=(input_dim)//4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),           
        )
        self.conv_b = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=(input_dim)//4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),             
        )

    def forward(self, x):
        x_c = x[:,:4]
        x_c = self.c_conv(x_c)
        
        x = torch.cat((x_c, x[:,4:]), dim=1)

        x_r = self.conv_r(x)
        x_g = self.conv_g(x)
        x_b = self.conv_b(x)
        x = torch.cat((x_r, x_g, x_b), dim=1)

        return x
    
class SteeperSigmoid(nn.Module):
    def __init__(self, temperature=0.1):
        super(SteeperSigmoid, self).__init__()
        self.temperature = temperature

    def forward(self, x):
        return torch.sigmoid(x / self.temperature)
    
class OccupancyField_cnn_v2(nn.Module):
    def __init__(self, in_features=128, in_c_features=3, hidden_features=32, out_features=1):
        super(OccupancyField_cnn_v2, self).__init__()

        if True:
            if True:
                self.c_conv = nn.Sequential(
                    nn.Conv3d(in_channels=in_c_features, out_channels=in_c_features*2, kernel_size=7, stride=1, padding=3, groups=in_c_features),
                    nn.Conv3d(in_channels=in_c_features*2, out_channels=in_c_features*2, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),

                    nn.Conv3d(in_channels=in_c_features*2, out_channels=in_c_features*4, kernel_size=5, stride=1, padding=2, groups=in_c_features*2),
                    nn.Conv3d(in_channels=in_c_features*4, out_channels=in_c_features*4, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),

                    nn.Conv3d(in_channels=in_c_features*4, out_channels=in_c_features*8, kernel_size=3, stride=1, padding=1, groups=in_c_features*4),
                    nn.Conv3d(in_channels=in_c_features*8, out_channels=in_c_features*8, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),

                    nn.Conv3d(in_channels=in_c_features*8, out_channels=in_c_features*8, kernel_size=3, stride=1, padding=1, groups=in_c_features*8),
                    nn.Conv3d(in_channels=in_c_features*8, out_channels=in_c_features*8, kernel_size=1, stride=1, padding=0),
                    nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                )
            else:
                self.c_conv = nn.Sequential(
                    nn.Conv3d(in_channels=in_c_features, out_channels=in_c_features+2, kernel_size=5, stride=1, padding=2),
                    #nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),

                    nn.Conv3d(in_channels=in_c_features+2, out_channels=in_c_features+2, kernel_size=3, stride=1, padding=1),
                    #nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),

                    nn.Conv3d(in_channels=in_c_features+2, out_channels=in_c_features+2, kernel_size=1, stride=1, padding=0),
                    nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),

                    #nn.Conv3d(in_channels=in_c_features*2, out_channels=in_c_features*4, kernel_size=3, stride=1, padding=1),
                    #nn.BatchNorm3d(in_c_features//2),
                    #nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                    #nn.ReLU(inplace=True),

                    nn.Conv3d(in_channels=in_c_features+2, out_channels=in_c_features+4, kernel_size=5, stride=1, padding=2),
                    #nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),

                    nn.Conv3d(in_channels=in_c_features+4, out_channels=in_c_features+4, kernel_size=3, stride=1, padding=1),
                    #nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),

                    nn.Conv3d(in_channels=in_c_features+4, out_channels=in_c_features+4, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm3d(in_c_features+4),
                    nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),   
                )
        elif True:
            self.c_conv = nn.Sequential(
                nn.Conv3d(in_channels=in_c_features, out_channels=in_c_features, kernel_size=5, stride=1, padding=2),
                #nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                nn.Conv3d(in_channels=in_c_features, out_channels=in_c_features, kernel_size=3, stride=1, padding=1),
                #nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                nn.Conv3d(in_channels=in_c_features, out_channels=in_c_features, kernel_size=1, stride=1, padding=0),
                nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                #nn.Conv3d(in_channels=in_c_features*2, out_channels=in_c_features*4, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm3d(in_c_features//2),
                #nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                #nn.ReLU(inplace=True),

                nn.Conv3d(in_channels=in_c_features, out_channels=in_c_features//2, kernel_size=5, stride=1, padding=2),
                #nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                nn.Conv3d(in_channels=in_c_features//2, out_channels=in_c_features//2, kernel_size=3, stride=1, padding=1),
                #nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                nn.Conv3d(in_channels=in_c_features//2, out_channels=1, kernel_size=1, stride=1, padding=0),
                #nn.BatchNorm3d(1),
                nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),   
            )
        else:
            self.c_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_c_features, out_channels=4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=hidden_features, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),     
            )

        if True:
            #input_dim = in_features//32+in_c_features*4
            #input_dim = 33+in_c_features+4
            #input_dim = 64+in_c_features+4
            #input_dim = 128//32+in_c_features*8
            #input_dim = 65+in_c_features*8
            input_dim = 64
            self.conv_1_3d = nn.Sequential(
                    nn.Conv3d(in_channels=input_dim, out_channels=input_dim//2, kernel_size=5, stride=1, padding=2),
                    #nn.BatchNorm3d(input_dim//2),
                    nn.ReLU(inplace=True),

                    nn.Conv3d(in_channels=input_dim//2, out_channels=input_dim//4, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),

                    nn.Conv3d(in_channels=input_dim//4, out_channels=input_dim//4, kernel_size=1, stride=1, padding=0),
                    nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),

                    #nn.Conv3d(in_channels=input_dim-in_c_features*2, out_channels=in_features//32, kernel_size=3, stride=1, padding=1),
                    #nn.BatchNorm3d(in_features//32),
                    #nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                    #nn.ReLU(inplace=True),

                    nn.Conv3d(in_channels=input_dim//4, out_channels=1, kernel_size=5, stride=1, padding=2),
                    #nn.BatchNorm3d(input_dim//2),
                    nn.ReLU(inplace=True),

                    nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),

                    nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
                    nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                    nn.Sigmoid(),  
                )
            input_dim = input_dim//4
            self.conv_2_3d = nn.Sequential(
                    nn.Conv3d(in_channels=input_dim, out_channels=input_dim//2, kernel_size=5, stride=1, padding=2),
                    #nn.BatchNorm3d(input_dim),
                    nn.ReLU(inplace=True),

                    nn.Conv3d(in_channels=input_dim//2, out_channels=input_dim//2, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),

                    nn.Conv3d(in_channels=input_dim//2, out_channels=input_dim//2, kernel_size=1, stride=1, padding=0),
                    nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),

                    #nn.Conv3d(in_channels=input_dim, out_channels=input_dim//2, kernel_size=3, stride=1, padding=1),
                    #nn.BatchNorm3d(input_dim//2),
                    #nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                    #nn.ReLU(inplace=True),

                    nn.Conv3d(in_channels=input_dim//2, out_channels=1, kernel_size=5, stride=1, padding=2),
                    #nn.BatchNorm3d(input_dim//2),
                    nn.ReLU(inplace=True),

                    nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),

                    nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
                    nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                    nn.Sigmoid(),  
                )
        elif True:
            input_dim = in_features+hidden_features
            self.conv_1 = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(input_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=input_dim, out_channels=in_features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            )
            input_dim = in_features+hidden_features
            self.conv_2 = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(input_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=input_dim, out_channels=in_features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            )
            input_dim = in_features+hidden_features
            self.conv_3 = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(input_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=input_dim, out_channels=out_features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1, padding=0),
                SteeperSigmoid(temperature=0.01),
            )
        '''
        input_dim = in_features-4+hidden_features
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=(input_dim)//4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//4, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),     
        )
        '''
    
    if True:   # depthwise conv, using feature dim for depth dimension
        def forward(self, x, x_c):
            #x_c = self.c_conv(x_c)   # (batch_size, 1, depth_res, height_res, width_res)
            
            x_c_0 = x_c.clone().detach()
            x_c_0 = (x_c_0 - x_c_0.min()) / (x_c_0.max() - x_c_0.min() + 1e-6)
            selected_images = x_c_0[0, 0, :6]  # This is now of shape [6, H, W]
            concatenated_images_0 = torch.cat(tuple(selected_images), dim=-1)
            selected_images = x_c_0[0, 0, 6:12]  # This is now of shape [6, H, W]
            concatenated_images_1 = torch.cat(tuple(selected_images), dim=-1)
            selected_images = x_c_0[0, 0, 12:18]  # This is now of shape [6, H, W]
            concatenated_images_2 = torch.cat(tuple(selected_images), dim=-1)
            selected_images = x_c_0[0, 0, -18:-12]  # This is now of shape [6, H, W]
            concatenated_images_3 = torch.cat(tuple(selected_images), dim=-1)
            selected_images = x_c_0[0, 0, -12:-6]  # This is now of shape [6, H, W]
            concatenated_images_4 = torch.cat(tuple(selected_images), dim=-1)
            selected_images = x_c_0[0, 0, -6:]  # This is now of shape [6, H, W]
            concatenated_images_5 = torch.cat(tuple(selected_images), dim=-1)
            concatenated_images = torch.cat((concatenated_images_0, concatenated_images_1, concatenated_images_2, concatenated_images_3,
                                            concatenated_images_4, concatenated_images_5), dim=-2)
            concatenated_images_np = concatenated_images.cpu().numpy()
            wandb.log({"x_c occupancy field features": [wandb.Image(concatenated_images_np, mode='L')]})

            #x = x.unsqueeze(1).view(x.shape[0], -1, x_c.shape[2], x.shape[2], x.shape[3])   # (batch_size, features/depth_res, depth_res, height_res, width_res)
            #x = x.view(x.shape[0], -1, x_c.shape[2], x.shape[2], x.shape[3])   # (batch_size, features/depth_res, depth_res, height_res, width_res)
            #x = x.view(x.shape[0], -1, x_c.shape[2], x.shape[2], x.shape[3])   # (batch_size, features/depth_res, depth_res, height_res, width_res)
            #x = torch.cat((x, x_c), dim=1)
            x = self.conv_1_3d(x)
            #x = torch.cat((x, x_c), dim=1)
            #x = self.conv_2_3d(x)

            #x = x.squeeze(1)
            x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])

            return x
    elif True:
        def forward(self, x, x_c):
            x_c = self.c_conv(x_c)   # (batch_size, 1, depth_res, height_res, width_res)
            x = self.i_conv(x)   # (batch_size, 1, height_res, width_res)
            x = x.unsqueeze(2).repeat(1, 1, x_c.size(2), 1, 1)   # (batch_size, depth_res, height_res, width_res)

            x_c_0 = x_c.clone().detach()
            x_c_0 = (x_c_0 - x_c_0.min()) / (x_c_0.max() - x_c_0.min() + 1e-6)
            selected_images = x_c_0[0, 0, :6]  # This is now of shape [6, H, W]
            concatenated_images_0 = torch.cat(tuple(selected_images), dim=-1)
            selected_images = x_c_0[0, 0, 6:12]  # This is now of shape [6, H, W]
            concatenated_images_1 = torch.cat(tuple(selected_images), dim=-1)
            selected_images = x_c_0[0, 0, 12:18]  # This is now of shape [6, H, W]
            concatenated_images_2 = torch.cat(tuple(selected_images), dim=-1)
            selected_images = x_c_0[0, 0, -18:-12]  # This is now of shape [6, H, W]
            concatenated_images_3 = torch.cat(tuple(selected_images), dim=-1)
            selected_images = x_c_0[0, 0, -12:-6]  # This is now of shape [6, H, W]
            concatenated_images_4 = torch.cat(tuple(selected_images), dim=-1)
            selected_images = x_c_0[0, 0, -6:]  # This is now of shape [6, H, W]
            concatenated_images_5 = torch.cat(tuple(selected_images), dim=-1)
            concatenated_images = torch.cat((concatenated_images_0, concatenated_images_1, concatenated_images_2, concatenated_images_3,
                                            concatenated_images_4, concatenated_images_5), dim=-2)
            concatenated_images_np = concatenated_images.cpu().numpy()
            wandb.log({"x_c occupancy field features": [wandb.Image(concatenated_images_np, mode='L')]})

            #x = x.unsqueeze(1).view(x.shape[0], -1, x_c.shape[2], x.shape[2], x.shape[3])   # (batch_size, features/depth_res, depth_res, height_res, width_res)
            #x = x.view(x.shape[0], -1, x_c.shape[2], x.shape[2], x.shape[3])   # (batch_size, features/depth_res, depth_res, height_res, width_res)
            x = torch.cat((x, x_c), dim=1)
            x = self.conv_1_3d(x)
            #x = torch.cat((x, x_c), dim=1)
            x = self.conv_2_3d(x)

            #x = x.squeeze(1)
            x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])

            return x
    elif True:
        def forward(self, x, x_c):
            x_c = self.c_conv(x_c)   # (batch_size, 1, depth_res, height_res, width_res)
            x_c = x_c.squeeze(1)   

            '''
            x = x.unsqueeze(1).view(x.shape[0], -1, x_c.shape[2], x.shape[2], x.shape[3])   # (batch_size, features/depth_res, depth_res, height_res, width_res)
            x = torch.cat((x, x_c), dim=1)
            x = self.conv_1_3d(x)
            x = torch.cat((x, x_c), dim=1)
            x = self.conv_2_3d(x)

            x = x.squeeze(1)
            '''

            with torch.no_grad():
                x_c_0 = x_c[0, :9].clone().detach().view(x_c.shape[2], -1, 3).cpu().numpy()
                wandb.log({"x_c occupancy field features": [wandb.Image(x_c_0)]})

            split_size = x.size(1) // x_c.size(1)
            x_split = torch.split(x, split_size, dim=1)
            x_c_split = torch.split(x_c, 1, dim=1)
            interleaved = [item for pair in zip(x_split, x_c_split) for item in pair]
            x = torch.cat(interleaved, dim=1)
            #x = torch.stack((x, x_c), dim=1)
            #x = x.view(x.size(0), -1, x.size(3), x.size(4))
            x = self.conv_1(x)

            split_size = x.size(1) // x_c.size(1)
            x_split = torch.split(x, split_size, dim=1)
            x_c_split = torch.split(x_c, 1, dim=1)
            interleaved = [item for pair in zip(x_split, x_c_split) for item in pair]
            x = torch.cat(interleaved, dim=1)
            #x = torch.stack((x, x_c), dim=1)
            #x = x.view(x.size(0), -1, x.size(3), x.size(4))
            x = self.conv_2(x)

            split_size = x.size(1) // x_c.size(1)
            x_split = torch.split(x, split_size, dim=1)
            x_c_split = torch.split(x_c, 1, dim=1)
            interleaved = [item for pair in zip(x_split, x_c_split) for item in pair]
            x = torch.cat(interleaved, dim=1)
            #x = torch.stack((x, x_c), dim=1)
            #x = x.view(x.size(0), -1, x.size(3), x.size(4))
            x = self.conv_3(x)
            '''
            x = torch.cat((x_c, x[:,4:]), dim=1)
            x = self.conv(x)
            '''

            return x
    else:
        def forward(self, x):
            x_c = x[:,:4]
            x_c = self.c_conv(x_c)
            
            x = torch.cat((x_c, x[:,4:]), dim=1)
            x = self.conv_1(x)
            x = torch.cat((x, x_c), dim=1)
            x = self.conv_2(x)
            x = torch.cat((x, x_c), dim=1)
            x = self.conv_3(x)
            '''
            x = torch.cat((x_c, x[:,4:]), dim=1)
            x = self.conv(x)
                '''

            return x

class downsample_layer(nn.Module):
    def __init__(self, in_channels, out_channels, dim, mode='mp'):
        super(downsample_layer, self).__init__()

        if mode == 'mp':
            if dim == 2:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
                self.down = nn.MaxPool2d(kernel_size=2, stride=2)
                self.dw = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                )
            elif dim == 3:
                self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
                self.down = nn.MaxPool3d(kernel_size=2, stride=2)
                self.dw = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
                    nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                )
        elif mode == 'ps':
            if dim == 2:
                self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
                self.down = nn.PixelUnshuffle(2)
                self.dw = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_channels*4, out_channels=in_channels*4, kernel_size=3, stride=1, padding=1, groups=out_channels),
                    nn.Conv2d(in_channels=in_channels*4, out_channels=in_channels*4, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                )
            elif dim == 3:
                self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
                self.down = PixelUnshuffle3d(2)
                self.dw = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Conv3d(in_channels=in_channels*8, out_channels=in_channels*8, kernel_size=3, stride=1, padding=1, groups=out_channels),
                    nn.Conv3d(in_channels=in_channels*8, out_channels=in_channels*8, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                )

    def forward(self, x):
        x = self.conv(x)
        x = self.down(x)
        x = self.dw(x)

        return x
    
class upsample_layer(nn.Module):
    def __init__(self, in_channels, out_channels, dim, mode='ct'):
        super(upsample_layer, self).__init__()

        if mode == 'ct':
            if dim == 3:
                self.up = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
                self.conv = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
                self.dw = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
                    nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                )
        elif mode == 'ps':
            self.up = PixelShuffle3d(2)
            self.conv = nn.Conv3d(in_channels=in_channels//8, out_channels=in_channels//8, kernel_size=3, stride=1, padding=1)
            self.dw = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=in_channels//8, out_channels=in_channels//8, kernel_size=3, stride=1, padding=1, groups=out_channels),
                nn.Conv3d(in_channels=in_channels//8, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
            )


    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.dw(x)

        return x

class OccupancyField_cnn_v3(nn.Module):
    def __init__(self, in_features=128, in_c_features=3, hidden_features=32, out_features=1):
        super(OccupancyField_cnn_v3, self).__init__()

        down_3d = 'mp'   # mp, ps
        down_2d = 'ps'   # mp, ps
        up = 'ps'   # us, ps
        self.skip_con = True

        self.c_downconv_1 = downsample_layer(in_channels=in_c_features, out_channels=in_c_features*8, dim=3, mode='ps')   # (bs, 24, 16, 192, 192) | (batch_size, 3*2, depth_res//2, height_res//2, width_res//2)
        self.c_downconv_2 = downsample_layer(in_channels=in_c_features*8, out_channels=in_c_features*16, dim=3, mode='mp')   # (bs, 12, 8, 96, 96) | (batch_size, 3*4, depth_res//4, height_res//4, width_res//4)
        self.c_downconv_3 = downsample_layer(in_channels=in_c_features*16, out_channels=in_c_features*32, dim=3, mode='mp')   # (bs, 24, 4, 48, 48) | (batch_size, 3*8, depth_res//8, height_res//8, width_res//8)
        self.c_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_c_features*32, out_channels=in_c_features*32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=in_c_features*32, out_channels=in_c_features*32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(in_c_features*32),
            nn.ReLU(inplace=True),
        )
        
        in_i_features = 3
        self.i_downconv_1 = downsample_layer(in_channels=in_i_features, out_channels=in_i_features*4, dim=2, mode='ps')   # (bs, 16, 192, 192) | (batch_size, 4*4, height_res//2, width_res//2)
        self.i_downconv_2 = downsample_layer(in_channels=in_i_features*4, out_channels=in_i_features*8, dim=2, mode='mp')   # (bs, 16, 96, 96) | (batch_size, 4*4, height_res//4, width_res//4)
        self.i_downconv_3 = downsample_layer(in_channels=in_i_features*8, out_channels=in_i_features*16, dim=2, mode='mp')   # (bs, 32, 48, 48) | (batch_size, 4*8, height_res//8, width_res//8)
        self.i_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_i_features*16, out_channels=in_i_features*32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),   

            nn.Conv2d(in_channels=in_i_features*32, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),   
        )

        #input_dim = 128//32+in_c_features*8
        #input_dim = 24+32   # mp_2d + mp_3d
        #input_dim = 256+in_c_features*8   # ps_2d + mp_3d
        #input_dim = 64+in_c_features*32   # ps_2d, mp_2d, mp_2d + ps_3d, mp_3d, mp_3d
        input_dim = 128+in_c_features*32   # ps_2d, mp_2d, mp_2d + ps_3d, mp_3d, mp_3d
        self.deconv_1 = nn.Sequential(
            nn.Conv3d(in_channels=input_dim, out_channels=in_c_features*32+in_i_features*32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),   

            nn.Conv3d(in_channels=in_c_features*32+in_i_features*32, out_channels=in_c_features*32+in_i_features*16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  
        )

        input_dim = in_c_features*32 + in_i_features*16
        if self.skip_con:
            input_dim = input_dim + in_c_features*32 + in_i_features*16
        else:
            input_dim = input_dim
        #input_dim = input_dim + in_c_features*32 + 64
        self.upconv_1 = upsample_layer(in_channels=input_dim, out_channels=input_dim//2, dim=3, mode='ct')
        if self.skip_con:
            res = (input_dim//4 + in_c_features*8 + in_i_features*4) % 8
            self.upconv_2 = upsample_layer(in_channels=input_dim//2, out_channels=input_dim//4+res, dim=3, mode='ct')
        else:
            self.upconv_2 = upsample_layer(in_channels=input_dim//2, out_channels=input_dim//4, dim=3, mode='ct')
        
        if self.skip_con:
            input_dim = input_dim//4 + in_c_features*8 + in_i_features*4 + res
        else:
            input_dim = input_dim//4
        #input_dim = input_dim//4 + in_c_features*8 + 16
        self.upconv_3 = upsample_layer(in_channels=input_dim, out_channels=input_dim//8, dim=3, mode='ps')
        self.deconv_2 = nn.Sequential(
            nn.Conv3d(in_channels=input_dim//8, out_channels=input_dim//8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=input_dim//8, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        
    
    def forward(self, x, x_c):
        x_c_ps = self.c_downconv_1(x_c)
        x_c = self.c_downconv_2(x_c_ps)
        x_c_ct = self.c_downconv_3(x_c)
        x_c = self.c_conv(x_c_ct)   # (batch_size, 1, depth_res, height_res, width_res)

        x_ps = self.i_downconv_1(x)
        x = self.i_downconv_2(x_ps)
        x_ct = self.i_downconv_3(x)
        x = self.i_conv(x_ct)   # (batch_size, 128, height_res, width_res)   |   depthwise conv, using feature dim for depth dimension
        
        if False:   # visualization
            x_c_0 = x_c.clone().detach()
            x_c_0 = (x_c_0 - x_c_0.min()) / (x_c_0.max() - x_c_0.min() + 1e-6)
            selected_images = x_c_0[0, 0, :6]  # This is now of shape [6, H, W]
            concatenated_images_0 = torch.cat(tuple(selected_images), dim=-1)
            selected_images = x_c_0[0, 0, 6:12]  # This is now of shape [6, H, W]
            concatenated_images_1 = torch.cat(tuple(selected_images), dim=-1)
            selected_images = x_c_0[0, 0, 12:18]  # This is now of shape [6, H, W]
            concatenated_images_2 = torch.cat(tuple(selected_images), dim=-1)
            selected_images = x_c_0[0, 0, -18:-12]  # This is now of shape [6, H, W]
            concatenated_images_3 = torch.cat(tuple(selected_images), dim=-1)
            selected_images = x_c_0[0, 0, -12:-6]  # This is now of shape [6, H, W]
            concatenated_images_4 = torch.cat(tuple(selected_images), dim=-1)
            selected_images = x_c_0[0, 0, -6:]  # This is now of shape [6, H, W]
            concatenated_images_5 = torch.cat(tuple(selected_images), dim=-1)
            concatenated_images = torch.cat((concatenated_images_0, concatenated_images_1, concatenated_images_2, concatenated_images_3,
                                            concatenated_images_4, concatenated_images_5), dim=-2)
            concatenated_images_np = concatenated_images.cpu().numpy()
            wandb.log({"x_c occupancy field features": [wandb.Image(concatenated_images_np, mode='L')]})

        x = x.unsqueeze(2).repeat(1, 1, x_c.size(2), 1, 1)   # (batch_size, n_features, depth_res_[enc], height_res_[enc], width_res_[enc])
        x = torch.cat((x, x_c), dim=1)
        x = self.deconv_1(x)

        if self.skip_con:
            x_ct = x_ct.unsqueeze(2).repeat(1, 1, x_c_ct.size(2), 1, 1)
            x = torch.cat((x, x_c_ct, x_ct), dim=1)
        x = self.upconv_1(x)
        x = self.upconv_2(x)
        
        if self.skip_con:
            x_ps = x_ps.unsqueeze(2).repeat(1, 1, x_c_ps.size(2), 1, 1)
            x = torch.cat((x, x_c_ps, x_ps), dim=1)
        x = self.upconv_3(x)
        x = self.deconv_2(x)

        #x = x.squeeze(1)   # (batch_size, depth_res, height_res, width_res)
        # If returning 4 channels, (batch_size, 4, depth_res, height_res, width_res) including rgb channels
        x_occ = x[:, -1]
        x_rgb = x[:, :-1]

        return x_occ.squeeze(1), x_rgb

class RGBfield_cnn_v2(nn.Module):
    def __init__(self, in_features=128, in_c_features=4, hidden_features=32, out_features=3):
        super(RGBfield_cnn_v2, self).__init__()

        if True:
            self.c_conv = nn.Sequential(
                nn.Conv3d(in_channels=in_c_features, out_channels=in_c_features, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm3d(in_c_features),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=in_c_features, out_channels=in_c_features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=in_c_features, out_channels=in_c_features//2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=in_c_features//2, out_channels=in_c_features//2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=in_c_features//2, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.AdaptiveMaxPool3d((hidden_features, 384, 384)),
                nn.ReLU(inplace=True)
            )
        elif False:
            self.c_conv = nn.Sequential(
                nn.Conv3d(in_channels=in_c_features, out_channels=in_c_features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=in_c_features, out_channels=in_c_features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=in_c_features, out_channels=in_c_features, kernel_size=1, stride=1, padding=0),
                nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=in_c_features, out_channels=in_c_features//2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=in_c_features//2, out_channels=in_c_features//2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=in_c_features//2, out_channels=in_c_features//2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=in_c_features//2, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),     
            )
        else:
            self.c_conv = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=hidden_features, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),     
            )

        input_dim = in_features+hidden_features
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((384, 384)),
            nn.Conv2d(in_channels=input_dim, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((384, 384)),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((384, 384)),
        )
        input_dim = in_features+hidden_features
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((384, 384)),
            nn.Conv2d(in_channels=input_dim, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((384, 384)),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((384, 384)),
        )
        input_dim = in_features+hidden_features
        self.conv_r = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((384, 384)),
            nn.Conv2d(in_channels=input_dim, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((384, 384)),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()         
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((384, 384)),
            nn.Conv2d(in_channels=input_dim, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((384, 384)),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()        
        )
        self.conv_b = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((384, 384)),
            nn.Conv2d(in_channels=input_dim, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((384, 384)),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()      
        )

    if True:
        def forward(self, x, x_c):
            x_c = self.c_conv(x_c)
            x_c = x_c.squeeze(1)

            split_size = x.size(1) // x_c.size(1)
            x_split = torch.split(x, split_size, dim=1)
            x_c_split = torch.split(x_c, 1, dim=1)
            interleaved = [item for pair in zip(x_split, x_c_split) for item in pair]
            x = torch.cat(interleaved, dim=1)
            #x = torch.stack((x, x_c), dim=1)
            #x = x.view(x.size(0), -1, x.size(3), x.size(4))
            x = self.conv_1(x)

            split_size = x.size(1) // x_c.size(1)
            x_split = torch.split(x, split_size, dim=1)
            x_c_split = torch.split(x_c, 1, dim=1)
            interleaved = [item for pair in zip(x_split, x_c_split) for item in pair]
            x = torch.cat(interleaved, dim=1)
            #x = torch.stack((x, x_c), dim=1)
            #x = x.view(x.size(0), -1, x.size(3), x.size(4))
            x = self.conv_2(x)

            split_size = x.size(1) // x_c.size(1)
            x_split = torch.split(x, split_size, dim=1)
            x_c_split = torch.split(x_c, 1, dim=1)
            interleaved = [item for pair in zip(x_split, x_c_split) for item in pair]
            x = torch.cat(interleaved, dim=1)
            #x = torch.stack((x, x_c), dim=1)
            #x = x.view(x.size(0), -1, x.size(3), x.size(4))

            x_r = self.conv_r(x)
            x_g = self.conv_g(x)
            x_b = self.conv_b(x)
            x = torch.cat((x_r, x_g, x_b), dim=1)

            return x

    else:
        def forward(self, x):
            x_c = x[:,:4]
            x_c = self.c_conv(x_c)
            
            x = torch.cat((x_c, x[:,4:]), dim=1)
            x = self.conv_1(x)
            x = torch.cat((x, x_c), dim=1)
            x = self.conv_2(x)
            x = torch.cat((x, x_c), dim=1)

            x_r = self.conv_r(x)
            x_g = self.conv_g(x)
            x_b = self.conv_b(x)
            x = torch.cat((x_r, x_g, x_b), dim=1)

            return x

class ShadowField_cnn(nn.Module):
    def __init__(self, in_features=128, out_channels=1):
        super(ShadowField_cnn, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//2, out_channels=in_features//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//2, out_channels=in_features//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//2, out_channels=in_features//2, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//2, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//4, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),     
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class dptEncoder(nn.Module):
    def __init__(self, pretrained_model, layers_to_drop=1, show_layers_keys=False):
        super().__init__()

        self.pretrained_model = pretrained_model

        if show_layers_keys:
            depth_head_dict = OrderedDict(self.pretrained_model.depth_head.named_children())
            scratch_dict = OrderedDict(self.pretrained_model.depth_head.scratch.named_children())
            output_conv2 = OrderedDict(self.pretrained_model.depth_head.scratch.output_conv2.named_children())
            print(output_conv2.keys())

        delattr(self.pretrained_model.depth_head.scratch.output_conv2, '0')
        delattr(self.pretrained_model.depth_head.scratch.output_conv2, '1')
        delattr(self.pretrained_model.depth_head.scratch.output_conv2, '2')
        delattr(self.pretrained_model.depth_head.scratch.output_conv2, '3')
    
    def forward(self, x):

        return self.pretrained_model(x)
    
# Main model: UnA-Gen -------------------------------------------------------------------------------------------------
class UnaGenModel(nn.Module):
    def __init__(self, opt, in_channels=3, features=128):
        super(UnaGenModel, self).__init__()
        
        self.opt = opt
        self.batch_size = None   # If None, inherits from batch
        self.img_size = None   
        self.original_size = None
        self.image_size = None
        self.intrinsics = None   
        self.smpl_params = None
        self.pose = None
        self.cam_loc = None

        self.n_training_rays = opt['n_training_rays']
        self.mapping_dim = opt['mapping_dim']
        self.occupancy_threshold = opt['occupancy_threshold']
        self.matrix_mapping = self.initialize_matrix_mapping(self.mapping_dim, 0.5).cuda()

        self.activity_occupancy = None
        self.activity_occupancy_rgb = None
        self.ao_count = None
        self.ao = None
        self.ao_rgb = None

        self.prov = False
        self.enc = False
        #self.template_feat = nn.Parameter(torch.ones((1, 1, 32, 384, 384), requires_grad=True)).cuda()

        if opt['visualize_stats'] != 'epoch_end':
            self.visualize_stats = opt['visualize_stats']

        if self.enc:
            # Unet encoder initialization
            if opt['encoder']['name'] == 'unet':
                self.encoder = UNetEncoder(in_channels=3, out_channels=features).cuda()
                if not opt['encoder']['train']:
                    self.encoder.eval()
                    for param in self.encoder.parameters():
                        param.requires_grad = False

            # CNN encoder initialization
            if opt['encoder']['name'] == 'cnn':
                self.CNNEncoder = CNNEncoder().cuda()
                if not opt['encoder']['train']:
                    self.CNNEncoder.eval()
                    for param in self.CNNEncoder.parameters():
                        param.requires_grad = False

        if self.opt['add_cano_coo'] and not self.opt['add_depth_features']:
            self.OccupancyField = OccupancyField_cnn_v3(in_features=features, in_c_features=3, hidden_features=self.opt['decoder']['depth_dep_res'], out_features=self.opt['decoder']['depth_dep_res']).cuda()
            if self.prov:
                self.RGBField = RGBfield_cnn_v2(in_features=features, in_c_features=3, hidden_features=self.opt['decoder']['depth_dep_res'], out_features=self.opt['decoder']['depth_dep_res']).cuda()      
            if self.opt['shadow_field']:      
                self.Shader = RGBfield_cnn(in_features=features, out_channels=self.opt['decoder']['depth_dep_res']*2).cuda()
        elif self.opt['add_depth_features'] and not self.opt['add_cano_coo']:
            self.OccupancyField = OccupancyField_cnn(in_features=features+1, out_channels=self.opt['decoder']['depth_dep_res']).cuda()
            self.RGBField = RGBfield_cnn(in_features=features+1, out_channels=self.opt['decoder']['depth_dep_res']).cuda()   
            if self.opt['shadow_field']:           
                self.Shader = RGBfield_cnn(in_features=features, out_channels=self.opt['decoder']['depth_dep_res']*2).cuda()
        elif self.opt['add_cano_coo'] and self.opt['add_depth_features']:
            self.OccupancyField = OccupancyField_cnn_v2(in_features=features+1, in_c_features=4, out_channels=self.opt['decoder']['depth_dep_res']).cuda()
            self.RGBField = RGBfield_cnn_v2(in_features=features+1, in_c_features=4, out_channels=self.opt['decoder']['depth_dep_res']).cuda()   
            if self.opt['shadow_field']:           
                self.Shader = RGBfield_cnn(in_features=features, out_channels=self.opt['decoder']['depth_dep_res']*2).cuda()
        else:
            self.OccupancyField = OccupancyField_cnn(in_features=features, out_channels=self.opt['decoder']['depth_dep_res']).cuda()
            self.RGBField = RGBfield_cnn(in_features=features, out_channels=self.opt['decoder']['depth_dep_res']).cuda()
            if self.opt['shadow_field']:  
                self.Shader = RGBfield_cnn(in_features=features, out_channels=self.opt['decoder']['depth_dep_res']*2).cuda()

        self.apply(self.weights_init)   # weights initialization

    def forward(self, inputs):
        # input is a dictionary containing the image, the smpl parameters, and other info 
        # it is of shape (batch_size, key_values) where key values is of different dimensions
        t0 = time.time()  
        image = inputs['masked_image']
        self.img_size = (image.shape[2], image.shape[3])
        self.batch_size = image.shape[0]
        self.original_size = (inputs['original_size'][0].cuda(), inputs['original_size'][1].cuda())
        if self.batch_size == 1:
            self.image_size = torch.tensor([image.shape[-2], image.shape[-1]]).cuda()
        else:
            self.image_size = torch.stack(inputs['image_size']).cuda()
        min_x, max_x = inputs['min_x'].cuda(), inputs['max_x'].cuda()   
        min_y, max_y = inputs['min_y'].cuda(), inputs['max_y'].cuda()
        batch_idx = inputs['batch_idx']
        num_samples = inputs['num_samples']
        if self.opt['visualize_stats'] == 'epoch_end':
            if batch_idx == num_samples - 1:
                self.visualize_stats = True
                wandb.log({"epoch": inputs['epoch']})
            else:
                self.visualize_stats = False
       
        depth_image = inputs['depth_image'].squeeze(1).cuda()

        gender = inputs['metadata']['gender']
        betas = inputs['betas']
        # scale = inputs['smpl_params'][:, 0].cuda()
        self.betas = inputs['betas'].cuda()
        self.gender = inputs['metadata']['gender']
        self.smpl_tfs = inputs['smpl_tfs'].cuda()
        self.smpl_outputs = inputs['smpl_outputs']

        self.pose = inputs["pose"].cuda()
        self.intrinsics = inputs['intrinsics'].cuda()
        self.smpl_params = inputs['smpl_params'].cuda()
        self.smpl_verts_cano = inputs['smpl_verts_cano'].cuda()
        smpl_tfs = inputs['smpl_tfs'].cuda()
        smpl_outputs = inputs['smpl_outputs']

        with torch.no_grad():
            # Matrix mapping refinement
            if self.opt['active_occupancy_refinement'] and (inputs['epoch']+1) % self.opt['active_occupancy_refinement_epochs'] == 0 and batch_idx == 0:
                print("Active occupancy refinement at epoch", inputs['epoch'])
                self.active_occupancy_refinement(mode='preserve')
                #confs['loss']['canonical_consistency_loss'] = True

            if self.opt['phase_push'] and (inputs['epoch']+1) % 15 == 0 and batch_idx == 0:
                print("Phase push - active occupancy refinement at epoch", inputs['epoch'])
                self.active_occupancy_refinement(mode='preserve')
        
            # Learn normalization transformation for the metric spaces
            # I <> D <> C   I: Image space, D: Dynamical space (metric), C: Canonical space (metric)
            #      |    |
            #      D    C   D: Dynamical space (normalized), C: Canonical space (normalized)
            # The metric spaces are used to establish the 1-1 correspondence by skinning
            # The normalized spaces are used to establish the 1-1 correspondence by the feature encoder and evaluate the networks
            self.cam_loc = get_cam(self.pose)   # These parameters are used throught many methods

            # The active voxels are mapped by the matrix_mapping (while in the canonical space)
            active_voxels = self.voxel_mapping(self.matrix_mapping)   # (n_voxels, 3) | Returns the active voxels coordinates in the canonical space

            if True:
                if self.visualize_stats:
                    self.visualize_voxels(active_voxels, output_file='active_voxels.png')

            canonical_voxels_coo = active_voxels.unsqueeze(0).repeat(self.batch_size, 1, 1)   # (batch_size, n_voxels, 3)

            dynamical_voxels_world_coo = self.dynamical_tfs_from_canonical(canonical_voxels_coo.clone(),   # (batch_size, n_voxels, 3)
                                                                           smpl_tfs, betas, gender, smpl_outputs, batch=True)
            
            if True:
                if self.visualize_stats:
                    try:
                        self.visualize_voxels(canonical_voxels_coo[0].clone(), add_point=self.cam_loc[0].clone(), output_file='world_canonical_voxels_coo.png', world=True)
                    except:
                        pass
            
            # uv scaling, get_uv returns values in original image space, we scale them to the current image space
            voxels_uv = get_uv(dynamical_voxels_world_coo, self.intrinsics, self.pose)   # (batch_size, n_voxels, 2)
            #D = inputs['D'].cuda().float()
            #K = inputs['K'].cuda().float()
            #RT = inputs['RT'].cuda().float()
            #voxels_uv = get_uv(dynamical_voxels_world_coo, K4x4, RT)   # (batch_size, n_voxels, 2)
            #voxels_uv = world_to_pixel_batch(dynamical_voxels_world_coo, K, RT)   # range is wrong
            #voxels_uv = undistort_points(voxels_uv, self.intrinsics, D, max_iterations=10, tolerance=1e-6)
            voxels_uv[..., 0] = (voxels_uv[..., 0] - min_x[:, None]) / (max_x[:, None] - min_x[:, None]) * torch.tensor(self.img_size[1], device=voxels_uv.device).repeat(self.batch_size)[:, None]
            voxels_uv[..., 1] = (voxels_uv[..., 1] - min_y[:, None]) / (max_y[:, None] - min_y[:, None]) * torch.tensor(self.img_size[0], device=voxels_uv.device).repeat(self.batch_size)[:, None]
            voxels_uv[..., 0] = torch.clamp(voxels_uv[..., 0], 0, self.img_size[1]-1)   # This supposes images have the same size, but it is a given that it is the case
            voxels_uv[..., 1] = torch.clamp(voxels_uv[..., 1], 0, self.img_size[0]-1)
            
            # Place grid in the dynamical space with canonical coordinates
            uvz_canonical, mask_cano_coo = self.cano_coo_in_dynamical_space(dynamical_voxels_world_coo,   # (batch_size, 3, depth_res, height, width)
                                                                            min_x, max_x, min_y, max_y)   # (batch_size, 1, depth_res, height, width)

            if self.opt['template_feat']:
                #template_feat = torch.ones_like(uvz_canonical[:, 0], device=uvz_canonical.device, requires_grad=True).unsqueeze(1)
                uvz_canonical = torch.cat((uvz_canonical, self.template_feat), dim=1)
                print("template_feat", self.template_feat.shape)
                print("template_feat range", self.template_feat.min(), self.template_feat.max())
                
            if True:
                if self.visualize_stats:
                    self.visualize_voxels(dynamical_voxels_world_coo[0].clone(), output_file='dynamical_voxels_world_coo.png', world=True)

        # ---
        # Trained unet encoder
        # The feature encoder produces a pixel/voxel-wise feature embedding used to compute the occupancy field of the active voxels
        if self.enc:
            if self.opt['encoder']['train']:   # The encoder is trained 
                if self.opt['encoder']['name'] == 'unet':
                        #image_info = torch.cat((image, depth_image.unsqueeze(1)), dim=1)
                        features = self.encoder(image)

                if self.opt['encoder']['name'] == 'cnn':
                    features = self.CNNEncoder(image)

            elif not self.opt['encoder']['train']:   # Training of the encoder is paused
                with torch.no_grad():   
                    if self.opt['encoder']['name'] == 'unet':
                        features = self.encoder(image)

                    if self.opt['encoder']['name'] == 'cnn':
                        features = self.CNNEncoder(image)
            
        # --- Add canonical coordinates to the features ---
        if self.opt['add_cano_coo']:
            if True:
                features_coo = torch.clamp(uvz_canonical, -1, 1)   # (self.batch_size, 3, depth_res, height, width)
            elif True:   # The idea here is, instead of backprojecting values to add to the features,
                       # produce an extra dimension feature tensor including canonical info, to be passed through the 3d convs in the networks
                with torch.no_grad():
                    t3dconv_0 = time.time()
                    height, width = image.shape[2], image.shape[3]
                    depth_res = self.opt['decoder']['depth_dep_res']
                    #features_coo = -torch.ones((self.batch_size, 3, depth_res, height, width), requires_grad=True).cuda() 
                    features_coo = torch.zeros((self.batch_size, 3, depth_res, height, width)).cuda() 
                    image_height, image_width = self.img_size  
                    u, v = voxels_uv[:, :, 0], voxels_uv[:, :, 1] 
                    u = u / image_width * (width - 1)
                    v = v / image_height * (height - 1)
                    u = torch.clamp(u, 0, width-1).round().long().unsqueeze(1).repeat(1, 3, 1)
                    v = torch.clamp(v, 0, height-1).round().long().unsqueeze(1).repeat(1, 3, 1)
                    depth_values = torch.norm(dynamical_voxels_world_coo - self.cam_loc.unsqueeze(1), dim=-1)
                    d_min = 0.5
                    d_max = 3.5
                    depth_values = (depth_values - d_min) / (d_max - d_min)   # This normalization needs to be in accordance with the features interpolation method
                    #depth_values = (depth_values - depth_values.min(dim=1, keepdim=True)[0]) / (depth_values.max(dim=1, keepdim=True)[0] - depth_values.min(dim=1, keepdim=True)[0] + 1e-6) 
                    z = (depth_values * (depth_res - 1)).round().long().unsqueeze(1).repeat(1, 3, 1)
                    batch_indices = torch.arange(self.batch_size).view(-1, 1, 1)
                    channels_indices = torch.arange(3).view(1, -1, 1)  
                    cano_info = canonical_voxels_coo.transpose(1, 2)
                    #cano_info = torch.cat((canonical_voxels_coo.transpose(1, 2), depth_values), dim=1)
                    features_coo[batch_indices, channels_indices, z, v, u] = cano_info

                    mask_cano_coo = torch.zeros((self.batch_size, 1, depth_res, height, width), dtype=torch.float32, device=features_coo.device)
                    mask_cano_coo[batch_indices, 0, z, v, u] = 1
                    #template_param = torch.randint(-1, 100, (self.batch_size, 1, depth_res, height, width), device=features_coo.device, requires_grad=True).float()
                    #features_coo = torch.cat((features_coo, template_param), dim=1)
                    
                    '''
                    image_mask = (image.sum(dim=1) > 0).float().unsqueeze(1).unsqueeze(1).repeat(1, 3, depth_res, 1, 1)
                    #features_coo = features_coo * image_mask + (1 - image_mask) # put 1s on the outside
                    
                    features_coo_mask = (features_coo == 0).float()
                    weights = (features_coo != 0).float()
                    #conv = nn.Conv3d(3, 3, kernel_size=3, stride=1, padding=1).cuda()
                    #conv.weight = nn.Parameter(torch.ones_like(conv.weight)/27)
                    conv = nn.Conv3d(3, 3, kernel_size=7, stride=1, padding=3, bias=False).cuda()
                    conv.weight = nn.Parameter(torch.ones_like(conv.weight)/343)
                    #iter = 0
                    #drift = 0
                    while features_coo_mask.any():
                        #iter += 1
                        #drift += 0.1
                        #drift = min(drift, 1)
                        features_coo_int = conv(features_coo)
                        weights_int = conv(weights) 
                        #weights_int = torch.max(torch.tensor(drift, device=weights_int.device), weights_int)
                        #features_coo = features_coo * (1 - weights)
                        #features_coo = features_coo / weights
                        update_mask = (weights_int > 0).float() * features_coo_mask
                        features_coo = torch.where(update_mask.bool(), features_coo_int / (weights_int + 1e-6), features_coo)
                        #features_coo = features_coo + ((features_coo_int * features_coo_mask) / (weights_int * features_coo_mask + 1e-6))
                        features_coo_mask = (features_coo == 0).float()
                        weights = (features_coo != 0).float()
                    #features_coo = features_coo * image_mask + (1 - image_mask) * torch.sign(features_coo)   # put 1s on the outside
                    features_coo = features_coo * image_mask - (1 - image_mask)   # put -1s on the outside
                    '''
                    features_coo = (features_coo + 1) / 2   # For the networks

                    t3dconv_1 = time.time()
                    wandb.log({"3d conv time": t3dconv_1 - t3dconv_0})

                    x_c_0 = (features_coo).clone().detach()

                    selected_images = x_c_0[0, 0, :6]  # This is now of shape [6, H, W]
                    concatenated_images_0 = torch.cat(tuple(selected_images), dim=-1)
                    selected_images = x_c_0[0, 0, 6:12]  # This is now of shape [6, H, W]
                    concatenated_images_1 = torch.cat(tuple(selected_images), dim=-1)
                    selected_images = x_c_0[0, 0, 12:18]  # This is now of shape [6, H, W]
                    concatenated_images_2 = torch.cat(tuple(selected_images), dim=-1)
                    selected_images = x_c_0[0, 0, -18:-12]  # This is now of shape [6, H, W]
                    concatenated_images_3 = torch.cat(tuple(selected_images), dim=-1)
                    selected_images = x_c_0[0, 0, -12:-6]  # This is now of shape [6, H, W]
                    concatenated_images_4 = torch.cat(tuple(selected_images), dim=-1)
                    selected_images = x_c_0[0, 0, -6:]  # This is now of shape [6, H, W]
                    concatenated_images_5 = torch.cat(tuple(selected_images), dim=-1)
                    concatenated_images = torch.cat((concatenated_images_0, concatenated_images_1, concatenated_images_2, concatenated_images_3,
                                                    concatenated_images_4, concatenated_images_5), dim=-2)
                    concatenated_images_np = concatenated_images.cpu().numpy()
                    wandb.log({"x_c features coo": [wandb.Image(concatenated_images_np, mode='L')]})

                    #features_coo = features_coo * 2 - 1   # For the networks

            else:
                height, width = features.shape[2], features.shape[3]
                features_coo = torch.zeros((self.batch_size, 4, height, width)).cuda()   # We also add the voxel depth to add dynamical space information
                voxels_depths = torch.norm(dynamical_voxels_world_coo - self.cam_loc.unsqueeze(1), dim=-1).unsqueeze(1)
                u, v = voxels_uv[:, :, 0], voxels_uv[:, :, 1]
                image_height, image_width = self.img_size
                u = u * width / image_width
                v = v * height / image_height
                u = torch.clamp(u, 0, width-1).round().long().unsqueeze(1).repeat(1, 4, 1)
                v = torch.clamp(v, 0, height-1).round().long().unsqueeze(1).repeat(1, 4, 1)
                batch_indices = torch.arange(self.batch_size).view(-1, 1, 1)
                channels_indices = torch.arange(4).view(1, -1, 1)        
                features_coo[batch_indices, channels_indices, v, u] = torch.cat((canonical_voxels_coo.detach().transpose(1, 2), voxels_depths), dim=1)
                features = torch.cat((features_coo, features), dim=1)

        if self.enc:
            if self.visualize_stats:
                with torch.no_grad():
                    features_0 = features[0, :9].clone().detach().view(features.shape[2], -1, 3).cpu().numpy()
                    wandb.log({'features': [wandb.Image(features_0)]})

        #if self.enc:
            #features = torch.cat((features, depth_image.unsqueeze(1)), dim=1)
        # --- Occupancy ---
        if True:
            if self.enc:
                features_expanded = features.unsqueeze(2).repeat(1, 1, self.opt['decoder']['depth_dep_res'], 1, 1)
            image_info = torch.cat((image, depth_image.unsqueeze(1)), dim=1)
            image_info = image_info*2 - 1
            #features_coo = features_coo.requires_grad_(True)
            occupancy_field_t, rgb_field_t = self.OccupancyField(image, features_coo)

            #occupancy_field_grad = torch.autograd.grad(outputs=occupancy_field_t[mask_cano_coo.squeeze(1)], inputs=features_coo[mask_cano_coo.repeat(1, 3, 1, 1, 1)],
            #                        grad_outputs=torch.ones_like(occupancy_field_t[mask_cano_coo.squeeze(1)], requires_grad=False, device=occupancy_field_t.device),
            #                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        else:
            occupancy_field_t = self.OccupancyField(features)
        #of_grad_t0 = time.time()
        #occupancy_field_grad = torch.autograd.grad(outputs=occupancy_field_t, inputs=features_coo,
        #                            grad_outputs=torch.ones_like(occupancy_field_t, requires_grad=False, device=occupancy_field_t.device),
        #                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        #of_grad_t1 = time.time()
        
        outliers_mask = (1 - (image.mean(dim=1) > 0).float())

        if self.visualize_stats:
            with torch.no_grad():
                occupancy_field_t_0 = occupancy_field_t.clone().detach()
                dynamical_voxels_world_coo_0 = dynamical_voxels_world_coo.clone().detach()
                depth_values_0 = torch.norm(dynamical_voxels_world_coo_0 - self.cam_loc.unsqueeze(1), dim=-1)
                z_min, z_max = depth_values_0.min(dim=1)[0].unsqueeze(1), depth_values_0.max(dim=1)[0].unsqueeze(1)
                z = (depth_values_0 - z_min) / (z_max - z_min) * (self.opt['decoder']['depth_dep_res'] - 1)
                z = z.round().long()
                occupancy_field_t_0 = occupancy_field_t_0.unsqueeze(-1).repeat(1, 1, 1, 1, 3)
                occupancy_field_t_0[..., :-1] = 0
                u, v = voxels_uv[:, :, 0].long(), voxels_uv[:, :, 1].long()
                occupancy_field_t_0[:, z, v, u, 0] = 1 

                inter = 8   # 8 for 64 depth_res, 6 for 32/36 depth_res
                selected_images = occupancy_field_t_0[0, :inter]  # This is now of shape [6, H, W]
                concatenated_images_0 = torch.cat(tuple(selected_images), dim=-2)
                selected_images = occupancy_field_t_0[0, inter:inter*2]  # This is now of shape [6, H, W]
                concatenated_images_1 = torch.cat(tuple(selected_images), dim=-2)
                selected_images = occupancy_field_t_0[0, inter*2:inter*3]  # This is now of shape [6, H, W]
                concatenated_images_2 = torch.cat(tuple(selected_images), dim=-2)
                selected_images = occupancy_field_t_0[0, -inter*3:-inter*2]  # This is now of shape [6, H, W]
                concatenated_images_3 = torch.cat(tuple(selected_images), dim=-2)
                selected_images = occupancy_field_t_0[0, -inter*2:-inter]  # This is now of shape [6, H, W]
                concatenated_images_4 = torch.cat(tuple(selected_images), dim=-2)
                selected_images = occupancy_field_t_0[0, -inter:]  # This is now of shape [6, H, W]
                concatenated_images_5 = torch.cat(tuple(selected_images), dim=-2)
                concatenated_images = torch.cat((concatenated_images_0, concatenated_images_1, concatenated_images_2, concatenated_images_3,
                                                 concatenated_images_4, concatenated_images_5), dim=0)
                concatenated_images_np = concatenated_images.cpu().numpy()

                x_c_0 = (features_coo).clone().detach()

                selected_images = x_c_0[0, 0, :inter]  # This is now of shape [6, H, W]
                concatenated_images_0 = torch.cat(tuple(selected_images), dim=-1)
                selected_images = x_c_0[0, 0, inter:inter*2]  # This is now of shape [6, H, W]
                concatenated_images_1 = torch.cat(tuple(selected_images), dim=-1)
                selected_images = x_c_0[0, 0, inter*2:inter*3]  # This is now of shape [6, H, W]
                concatenated_images_2 = torch.cat(tuple(selected_images), dim=-1)
                selected_images = x_c_0[0, 0, -inter*3:-inter*2]  # This is now of shape [6, H, W]
                concatenated_images_3 = torch.cat(tuple(selected_images), dim=-1)
                selected_images = x_c_0[0, 0, -inter*2:-inter]  # This is now of shape [6, H, W]
                concatenated_images_4 = torch.cat(tuple(selected_images), dim=-1)
                selected_images = x_c_0[0, 0, -inter:]  # This is now of shape [6, H, W]
                concatenated_images_5 = torch.cat(tuple(selected_images), dim=-1)
                concatenated_images = torch.cat((concatenated_images_0, concatenated_images_1, concatenated_images_2, concatenated_images_3,
                                                concatenated_images_4, concatenated_images_5), dim=-2)
                concatenated_images_np_x_c = concatenated_images.cpu().numpy()
                wandb.log({"x_c features coo": [wandb.Image(concatenated_images_np_x_c, mode='L'), wandb.Image(concatenated_images_np)]})

        weights = torch.linspace(1, 0, steps=occupancy_field_t.shape[1], device=occupancy_field_t.device)
        weights = weights.view(1, -1, 1, 1)
        cum_of = torch.clamp(torch.cumsum(occupancy_field_t, dim=1) - occupancy_field_t, 0, 1)
        of_dpt = (occupancy_field_t * weights * (1 - cum_of)).sum(dim=1)   

        logits = (-((1 + 10*cum_of) * (1 + (1 - 10*occupancy_field_t))))*100   # *100 ?
        #logits = (-(torch.exp(cum_of) * torch.exp(1 - occupancy_field_t)))*100   # *100 ?
        #logits = (-(cum_of * occupancy_field_t) - (1 - occupancy_field_t)*occupancy_field_t.shape[1])*100   # *100 ?
        softmin = stable_softmax(logits, dim=1)
        #front_voxels = softmin * occupancy_field_t
        front_voxels = softmin
        
        voxel_idxs = torch.linspace(1, 0, steps=softmin.shape[1], device=softmin.device).view(1, -1, 1, 1).repeat(self.batch_size, 1, softmin.shape[2], softmin.shape[3])
        of_dpt_e = (voxel_idxs * softmin).sum(dim=1)
        of_dpt_e = of_dpt_e * (image.mean(dim=1) > 0).float()
        min_val, max_val = torch.amin(of_dpt_e, dim=(-1, -2)), torch.amax(of_dpt_e, dim=(-1, -2))
        of_dpt_e = of_dpt_e / (max_val.view(-1, 1, 1) + 1e-6)

        if self.visualize_stats:   # vis of_dpt
            stacked_images = torch.cat((of_dpt.clone().detach()[0], depth_image.clone().detach()[0]), dim=1)
            wandb.log({'of_dpt': [wandb.Image(stacked_images.clone().detach().cpu().numpy(), mode='L')]})
        if self.visualize_stats:   # vis of_dpt_e
            stacked_images = torch.cat((of_dpt_e.clone().detach()[0], depth_image.clone().detach()[0]), dim=1)
            wandb.log({'of_dpt_e': [wandb.Image(stacked_images.clone().detach().cpu().numpy(), mode='L')]})

        occupancy_field = self.interpolate_to_voxels(occupancy_field_t, dynamical_voxels_world_coo, voxels_uv)

        if self.visualize_stats:   # vis soft_min
            non_zero_idxs = softmin > 0.01
            wandb.log({'softmin': wandb.Histogram(softmin[non_zero_idxs].view(-1).clone().detach().cpu().numpy())})

        if self.visualize_stats:   # vis occupancy_field values distribution
            occupancy_field_0 = occupancy_field[0].view(-1).clone().detach().cpu().numpy()
            with torch.no_grad():
                wandb.log({"Occupancy field values distribution": wandb.Histogram(occupancy_field_0)})

                if False:
                    if self.visualize_stats:
                        occupancy_map = (occupancy_field[0] > self.occupancy_threshold).squeeze(-1)
                        occupied_world_canonical_voxels_coo = canonical_voxels_coo[0][occupancy_map]
                        self.visualize_voxels(occupied_world_canonical_voxels_coo, output_file='occupied_world_canonical_voxels_coo.png', world=True)

        # --- Rendering ---
        if self.prov:
            if True:
                rgb_field_t = self.RGBField(features, features_coo)
            else:
                rgb_field_t = self.RGBField(features)
            if self.opt['shadow_field']:  
                shadow_field_t = self.Shader(features[:, 4:-1])   # This has to change depending how much was attachhed in front of the features
                voxels_rgb_t = shadow_field_t[:, :(shadow_field_t.shape[1]//2)] + shadow_field_t[:, (shadow_field_t.shape[1]//2):] * rgb_field_t
                #voxels_rgb_t = shadow_field_t * rgb_field_t
                #voxels_rgb_t = torch.cat([shadow_field_t, shadow_field_t, shadow_field_t], dim=1) * rgb_field_t
            else:
                voxels_rgb_t = rgb_field_t
            #rgb_field_t = torch.zeros(self.batch_size, 3, rgb_field.shape[1]//3, rgb_field.shape[2], rgb_field.shape[3]).cuda()
            #depth_res = rgb_field_t.shape[1]//3
            #for i in range(3):
            #    rgb_field_t[:, i] = rgb_field[:, i*depth_res:(i+1)*depth_res]
            rgb_field = self.interpolate_to_voxels(rgb_field_t, dynamical_voxels_world_coo, voxels_uv, rgb=True)
            voxels_rgb = self.interpolate_to_voxels(voxels_rgb_t, dynamical_voxels_world_coo, voxels_uv, rgb=True)
            rgb_field_t = rgb_field_t.view(self.batch_size, 3, rgb_field_t.shape[1]//3, rgb_field_t.shape[2], rgb_field_t.shape[3])
            voxels_rgb_t = voxels_rgb_t.view(self.batch_size, 3, voxels_rgb_t.shape[1]//3, voxels_rgb_t.shape[2], voxels_rgb_t.shape[3])
            
            #rasterized_image = (front_voxels.unsqueeze(1).expand_as(rgb_field_t) * rgb_field_t).sum(dim=2)
            rasterized_image = (front_voxels.unsqueeze(1).expand_as(voxels_rgb_t) * voxels_rgb_t).sum(dim=2)
            rasterized_image = torch.clamp(rasterized_image, 0, 1)
            stacked_images = torch.cat((rasterized_image.clone().detach()[0], image.clone().detach()[0]), dim=1)
            wandb.log({'rasterized_image': [wandb.Image(stacked_images.clone().detach().cpu().numpy(), mode='L')]})

            if self.visualize_stats:
                occupancy_map = (occupancy_field.clone().detach() > 0.5).squeeze(-1)
                occ_cano_voxels_coo = canonical_voxels_coo.clone().detach()[occupancy_map]
                occ_cano_voxels_ov = occupancy_field.clone().detach()[occupancy_map]
                occ_cano_voxels_rgb = rgb_field.clone().detach()[occupancy_map]
                if False:
                    with torch.no_grad():
                        try:                    
                            self.visualize_voxels(occ_cano_voxels_coo.clone(), output_file='cano_voxels_rgb.png', world=True, voxels_ov=occ_cano_voxels_ov, voxels_rgb=occ_cano_voxels_rgb)
                        except:
                            pass
        else:
            rgb_field_t = rgb_field_t   # Form occupancy field network
            #rgb_field_t = image.unsqueeze(2).repeat(1, 1, self.opt['decoder']['depth_dep_res'], 1, 1)
            voxels_rgb_t = rgb_field_t
            rgb_field = self.interpolate_to_voxels(rgb_field_t, dynamical_voxels_world_coo, voxels_uv, rgb=True)
            #voxels_rgb = self.interpolate_to_voxels(voxels_rgb_t, dynamical_voxels_world_coo, voxels_uv, rgb=True)
            #rgb_field = torch.zeros_like(occupancy_field).repeat(1, 1, 3)
            voxels_rgb = rgb_field   # Not used at the moment
            rasterized_image = (front_voxels.unsqueeze(1).expand_as(voxels_rgb_t) * voxels_rgb_t).sum(dim=2)
            rasterized_image = torch.clamp(rasterized_image, 0, 1)
            stacked_images = torch.cat((rasterized_image.clone().detach()[0].permute(1,2,0), image.clone().detach()[0].permute(1,2,0)), dim=1)
            wandb.log({'rasterized_image': [wandb.Image(stacked_images.clone().detach().cpu().numpy())]})
        #shadow_field = self.Shader(features[:, 4:])
        #shadow_field = self.interpolate_to_voxels(shadow_field, dynamical_voxels_world_coo, voxels_uv)
        
        #voxels_rgb = torch.clamp(rgb_field * shadow_field.expand_as(rgb_field), 0, 1)   # Apply shadowing to the rgb field
                                                                     # voxels_rgb is the final rgb field to be rendered
                                                                     # rgb_field is the actual rgb field without shadowing

        #self.shadow_field = shadow_field   # TODO: remove this, now used only for visualization, but it wastes memory
        #voxels_rgb = rgb_field 
        # ---
        if self.visualize_stats:
            rgb_field_0 = rgb_field[0].clone().detach().cpu().numpy()
            voxels_rgb_0 = voxels_rgb[0].clone().detach().cpu().numpy()
            with torch.no_grad():
                wandb.log({"RGB field values distribution": wandb.Histogram(rgb_field_0)})

                #shadow_field_0 = shadow_field[0].clone().detach().cpu().numpy()
                #wandb.log({"Shadow field values distribution": wandb.Histogram(shadow_field_0)})
                
                wandb.log({"Voxels RGB values distribution": wandb.Histogram(voxels_rgb_0)})
        
        # --- Render values for optimization ---
        t0_render_rgb_values = time.time()
        if self.opt['n_training_rays'] > 0:
            cum_of_s = self.interpolate_to_voxels(cum_of, dynamical_voxels_world_coo, voxels_uv)
            ray_caster = RayCaster(self, dynamical_voxels_world_coo, voxels_uv, occupancy_field, cum_of_s, voxels_rgb, image, depth_image)
            training_values = ray_caster.render_values_at_rays(mode='training')
        t1_render_rgb_values = time.time()

        if self.visualize_stats:
            wandb.log({"Time for rendering rgb values": (t1_render_rgb_values - t0_render_rgb_values)})

        self.update_activity_occupancy(dynamical_voxels_world_coo, voxels_uv, softmin, 
                                       occupancy_field, rgb_field, inputs['epoch'], batch_idx, num_samples, mode='update')
        if False:   # Flag when the model should update immediately after first epoch the activity occupancy. For evaluation and refinement
            self.update_activity_occupancy(self, dynamical_voxels_world_coo, voxels_uv, softmin, 
                                           occupancy_field, rgb_field, inputs['epoch'], batch_idx, num_samples, mode='retrieve_matrix_mapping')
        
        outputs = {'image': image,
                   'dynamical_voxels_coo': dynamical_voxels_world_coo,
                   'occupancy_field': occupancy_field,
                   'occupancy_field_t': occupancy_field_t,
                   'mask_cano_coo': mask_cano_coo,
                   'softmin': softmin,
                   'outliers_mask': outliers_mask,
                   #'occupancy_field_grad': occupancy_field_grad,
                   #'cum_of': cum_of,
                   'of_dpt': of_dpt,
                   'of_dpt_e': of_dpt_e,
                   #'of_diff': of_diff,
                   'depth_image': depth_image,
                   'rgb_field': rgb_field, 
                   'rgb_field_t': rgb_field_t,   # This might not be needed when shadow field is present
                   'voxels_rgb_t': voxels_rgb_t,
                   'rasterized_image': rasterized_image,
                   #'shadow_field': shadow_field,
                   'activity_occupancy': self.activity_occupancy,
                   'activity_occupancy_rgb': self.activity_occupancy_rgb,
                   'epoch': inputs['epoch']}
        if self.opt['voxel_splatting']:
            outputs['original_rgb_values'] = training_values['original_rgb_values']
            outputs['rendered_rgb_values'] = training_values['rendered_rgb_values']
        if self.opt['ray_cast_rgb']:
            outputs['original_rgb_values_add'] = training_values['original_rgb_values_rays']
            outputs['rendered_rgb_values_add'] = training_values['rendered_rgb_values_rays']
        if self.opt['ray_cast_depth']:
            outputs['depth_values'] = training_values['depth_values']
            outputs['estimated_depth_values'] = training_values['estimated_depth_values']
        if self.opt['mask_pruning']:
            outputs['occupied_pixels_mask'] = training_values['occupied_pixels_mask']

        t1 = time.time()

        if self.visualize_stats:
            wandb.log({"Time for forward pass": t1-t0})

        #self.generate_mesh(canonical_voxels_coo, occupancy_field, rgb_field)
        
        return outputs

    # Active voxels refinement --------------------------------------------------------------------------------------------
    def active_occupancy_refinement(self, mode='full', sf=1.5, update=True):
        '''This method is used to refine the active voxels based on the occupancy field.
        '''
        matrix_mapping = self.matrix_mapping   # (mapping_dim, mapping_dim, mapping_dim)
        active_voxels_coo = self.voxel_mapping(matrix_mapping)   # (n_voxels, 3)
        n_active_voxels_prev = active_voxels_coo.shape[0]
        print("active_voxels_coo shape: ", active_voxels_coo.shape)
        print("number of active voxels before refinement: ", active_voxels_coo.shape[0])
        activity_occupancy = self.activity_occupancy   # (n_voxels, 1)
        print("activity_occupancy shape: ", activity_occupancy.shape)
        #thr_mean = torch.quantile(self.activity_occupancy, 0.1)
        #ao_map = (activity_occupancy > min(self.opt['ao_threshold'], thr_mean)).float().squeeze(-1)
        ao_map = (activity_occupancy > self.opt['ao_threshold']).float().squeeze(-1)
        print("ao_map shape: ", ao_map.shape)
        occupied_matrix_mapping = self.voxel_mapping(matrix_mapping, active_voxels_coo, ao_map)
        n_active_voxels_post = occupied_matrix_mapping.sum()
        print("occupied_matrix_mapping shape: ", occupied_matrix_mapping.shape)
        print("number of active voxels from activity occupancy: ", occupied_matrix_mapping.sum())
        
        # 3D dilation
        #occ_mm_np = occupied_matrix_mapping.cpu().numpy()
        #occ_mm_np = binary_dilation(occ_mm_np, structure=np.ones((3,3,3)))
        #occupied_matrix_mapping = torch.from_numpy(occ_mm_np).float().cuda()

        if mode == 'preserve':
            occ_mm_np = occupied_matrix_mapping.cpu().numpy()
            occ_mm_np = binary_dilation(occ_mm_np, structure=np.ones((3,3,3)), iterations=4)

            occ_mm_np = binary_erosion(occ_mm_np, structure=np.ones((3,3,3)), iterations=3)
            occupied_matrix_mapping = torch.from_numpy(occ_mm_np).float().cuda()

            # Increase resolution
            sf = (n_active_voxels_prev / n_active_voxels_post)**(1/3)
            print("scale factor: ", sf)
            occupied_matrix_mapping = upsample_matrix(occupied_matrix_mapping, scale_factor=sf.item(), threshold=0.5)
            print("number of active voxels after refinement: ", occupied_matrix_mapping.sum())

        if mode == 'reduce':
            occ_mm_np = occupied_matrix_mapping.cpu().numpy()
            occ_mm_np = binary_dilation(occ_mm_np, structure=np.ones((3,3,3)), iterations=2)

            occ_mm_np = binary_erosion(occ_mm_np, structure=np.ones((3,3,3)), iterations=2)
            occupied_matrix_mapping = torch.from_numpy(occ_mm_np).float().cuda()

        if mode == 'full':
            occ_mm_np = occupied_matrix_mapping.cpu().numpy()
            occ_mm_np = binary_dilation(occ_mm_np, structure=np.ones((5,5,5)))

            occ_mm_np = binary_erosion(occ_mm_np, structure=np.ones((5,5,5)))
            occupied_matrix_mapping = torch.from_numpy(occ_mm_np).float().cuda()

            # Increase resolution
            occupied_matrix_mapping = upsample_matrix(occupied_matrix_mapping, scale_factor=sf, threshold=0.4)

        if mode == 'hollow':
            occ_mm_np = occupied_matrix_mapping.cpu().numpy().astype(np.bool)
            print("number of active voxels before hollowing: ", occ_mm_np.sum())
            #occ_mm_np = binary_dilation(occ_mm_np, structure=np.ones((3,3,3)))
            #print("number of active voxels after dilation: ", occ_mm_np.sum())
            out_map = ~occ_mm_np
            occ_mm_np = binary_erosion(occ_mm_np, structure=np.ones((3,3,3)), iterations=1)
            print("number of active voxels after erosion: ", occ_mm_np.sum())
            inv_map = out_map + occ_mm_np
            print("number of active voxels after hollowing: ", inv_map.sum())
            occ_mm_np = ~inv_map 
            print("number of active voxels after inversion: ", occ_mm_np.sum())
            occupied_matrix_mapping = torch.from_numpy(occ_mm_np).float().cuda()
            
            occupied_matrix_mapping = upsample_matrix(occupied_matrix_mapping, scale_factor=1.5, threshold=0.4)
            #upsampled_matrix_mapping = occupied_matrix_mapping

        if update:
            self.matrix_mapping = occupied_matrix_mapping
            self.mapping_dim = occupied_matrix_mapping.shape[0]

        print("upsampled_matrix_mapping.shape: ", occupied_matrix_mapping.shape)
        print("number of active voxels after refinement: ", occupied_matrix_mapping.sum())

        return
    
    # Auxiliary methods ---------------------------------------------------------------------------------------------------
    def initialize_matrix_mapping(self, mmap_dim, mmap_res, mode='smpl'):
        '''This method is used to initialize the matrix_mapping with random positions close to the center.
        '''
        sample_rand = False
        N_voxels = 10000
        matrix_mapping = torch.ones(mmap_dim, mmap_dim, mmap_dim).cuda()
        active_voxels = self.voxel_mapping(matrix_mapping)
        
        if mode == 'smpl':   # Load info for one frame to obtain canonical inliers and outliers
            smplx=False
            if smplx:
                canonical_init_video_folder = '/home/lbocchi/UnA-Gen/data/data/train/0012_09/camera_1'
                smplx_data_path = '/home/lbocchi/UnA-Gen/data/data/train/0012_09/smplx.npy'
                betas = torch.tensor(np.load(smplx_data_path, allow_pickle=True).item()['betas'], dtype=torch.float32)[0].unsqueeze(0)
            else:
                canonical_init_video_folder = '/home/lbocchi/UnA-Gen/data/data/train/courtyard_laceShoe_00'
                betas_path = os.path.join(canonical_init_video_folder, "mean_shape.npy")
                betas = torch.tensor(np.load(betas_path)[None], dtype=torch.float32).cuda()
            smpl_tfs_dir = os.path.join(canonical_init_video_folder, "smpl_tfs")

            smpl_tfs_files = os.listdir(smpl_tfs_dir)
            random_index = random.choice(range(len(smpl_tfs_files)))
            
            smpl_tfs_path = os.path.join(smpl_tfs_dir, os.listdir(smpl_tfs_dir)[random_index])
            smpl_tfs = torch.load(smpl_tfs_path).cuda()
            smpl_verts_cano_dir = os.path.join(canonical_init_video_folder, "smpl_verts_cano")
            smpl_verts_cano_path = os.path.join(smpl_verts_cano_dir, os.listdir(smpl_verts_cano_dir)[random_index])
            smpl_verts_cano = torch.load(smpl_verts_cano_path).cuda()
            if smplx:
                metadata_path = os.path.join('/home/lbocchi/UnA-Gen/data/data/train/0012_09/', "metadata.yaml")
            else:
                metadata_path = os.path.join(canonical_init_video_folder, "metadata.yaml")
            with open(metadata_path, 'r') as file:
                metadata = yaml.safe_load(file)
            gender = metadata['gender']

            # Obtain canonical inliers and compute corresponding matrix mapping 
            deformer = SMPLDeformer(betas=betas, gender=gender)
            #smpl_verts_cano = smpl_verts_cano * 1.1
            smpl_verts_cano = smpl_verts_cano
            #x_d, outlier_mask = deformer.forward(active_voxels, smpl_tfs, return_weights=False, inverse=False, smpl_verts=smpl_verts_cano.unsqueeze(0))
            print("active_voxels shape: ", active_voxels.shape)
            print("smpl_tfs shape: ", smpl_tfs.shape)
            x_d, outlier_mask = deformer.forward(active_voxels, smpl_tfs, return_weights=False, inverse=False, smpl_verts=smpl_verts_cano.unsqueeze(0))
            ao_map = (~outlier_mask).float()
            if sample_rand:   # Random selection of active voxels
                valid_indices = torch.where(ao_map == 1)[0]
                idx = torch.randperm(valid_indices.shape[0])[:N_voxels]
                rand_mask = torch.zeros_like(ao_map)
                rand_mask[valid_indices[idx]] = 1
                ao_map = ao_map * rand_mask
            occupied_matrix_mapping = self.voxel_mapping(matrix_mapping, active_voxels, ao_map)   # The matrix mapping is initialized with the occupied voxels
        elif mode == 'saved':   # Load saved matrix mapping from previous training 
            occupied_matrix_mapping = torch.load('/home/lbocchi/UnA-Gen/data/data/train/courtyard_laceShoe_00/occupied_matrix_mapping.pth').cuda()
        else:
            occupied_matrix_mapping = matrix_mapping

        # Hollow the volume 
        if False:
            occ_mm_np = occupied_matrix_mapping.cpu().numpy().astype(np.bool)
            print("number of active voxels before hollowing: ", occ_mm_np.sum())
            #occ_mm_np = binary_dilation(occ_mm_np, structure=np.ones((3,3,3)))
            #print("number of active voxels after dilation: ", occ_mm_np.sum())
            out_map = ~occ_mm_np
            occ_mm_np = binary_erosion(occ_mm_np, structure=np.ones((5,5,5)))
            print("number of active voxels after erosion: ", occ_mm_np.sum())
            inv_map = out_map + occ_mm_np
            print("number of active voxels after hollowing: ", inv_map.sum())
            occ_mm_np = ~inv_map 
            print("number of active voxels after inversion: ", occ_mm_np.sum())
            occupied_matrix_mapping = torch.from_numpy(occ_mm_np).float().cuda()            

        # Perform dilation
        #occ_mm_np = occupied_matrix_mapping.cpu().numpy()
        #occ_mm_np = binary_dilation(occ_mm_np, structure=np.ones((3,3,3)))
        #occupied_matrix_mapping = torch.from_numpy(occ_mm_np).float().cuda()
        
        return occupied_matrix_mapping
    
    def update_activity_occupancy(self, dynamical_voxels_world_coo, voxels_uv, softmin, occupancy_field, rgb_field, epoch, batch_idx, num_samples, mode='update'):
        depth_all = torch.norm(dynamical_voxels_world_coo - self.cam_loc.unsqueeze(1), dim=-1)
        depth_all = (depth_all - depth_all.min(dim=1, keepdim=True)[0]) / (depth_all.max(dim=1, keepdim=True)[0] - depth_all.min(dim=1, keepdim=True)[0])
        depth_mask = (depth_all < 0.5).float()   # Only close enough voxels
        #activity_mask = torch.ones_like(activity_mask)
        #activity_mask = (depth_all <= 0.5).float().unsqueeze(-1)
        #activity_mask = ((1 - depth_all >= 0.5).float() * ((1 - depth_all)**2)).unsqueeze(-1)
        soft_vox = self.interpolate_to_voxels(softmin.detach(), dynamical_voxels_world_coo, voxels_uv)
        #soft_vox_mask = (soft_vox > 0.01).float() 
        soft_vox_mask = soft_vox 
        #activity_mask = (((1 - depth_all)*depth_mask)**2).unsqueeze(-1) 
        activity_mask = (((1 - depth_all))**4).unsqueeze(-1) 
        activity_mask = activity_mask + soft_vox_mask
        if epoch == 0 or (epoch+1) % self.opt['active_occupancy_refinement_epochs'] == 0 or self.opt['phase_push']:
            self.activity_occupancy = occupancy_field.detach().mean(dim=0)
            self.activity_occupancy_rgb = rgb_field.detach().mean(dim=0)
            self.opt['phase_push'] = False
        if self.ao is None:
            self.ao_count = activity_mask.sum(dim=0)
            self.ao = (occupancy_field.detach() * activity_mask).sum(dim=0) 
            self.ao_rgb = (rgb_field.detach() * activity_mask).sum(dim=0) 
        else:
            # Contribute to activity occupancy and ao_rgb only with values towards the front, se we correct the further values
            self.ao_count += activity_mask.sum(dim=0)
            self.ao += (occupancy_field.detach() * activity_mask).sum(dim=0) 
            self.ao_rgb += (rgb_field.detach() * activity_mask).sum(dim=0) 
            if batch_idx == num_samples - 1:   # On epoch end average the occupancy values
                #self.activity_occupancy = self.ao / (self.batch_size * inputs['num_samples'])
                self.activity_occupancy = self.ao / (self.ao_count + 1e-6)
                self.activity_occupancy_rgb = self.ao_rgb / (self.ao_count + 1e-6)
                self.ao = None
                self.ao_rgb = None
                self.ao_count = None
                wandb.log({"activity_occupancy values distribution": wandb.Histogram(self.activity_occupancy.clone().detach().cpu().numpy())})                 
        
        if mode == 'retrieve_matrix_mapping':   # used for evaluation after first epoch. It does not refine, it simply immedaitely updates based on predictions
            occupied_voxels = self.activity_occupancy > self.opt['ao_threshold']
            matrix_mapping = self.matrix_mapping
            active_voxels_coo = self.voxel_mapping(matrix_mapping)
            self.matrix_mapping = self.voxel_mapping(matrix_mapping, active_voxels_coo, occupied_voxels)

        return

    def update_matrix_mapping(self, matrix_mapping, occupancy_field):
        '''This method is used to update the matrix_mapping based on the occupancy field.
        '''
        return matrix_mapping

    def cano_coo_in_dynamical_space(self, dynamical_voxels_world_coo, min_x, max_x, min_y, max_y):
        image_height = self.img_size[0]
        image_width = self.img_size[1]
        depth_dep_res = self.opt['decoder']['depth_dep_res']         
        u = torch.linspace(0, 1, image_width, device=dynamical_voxels_world_coo.device).view(1, 1, image_width).repeat(self.batch_size, 1, 1)
        v = torch.linspace(0, 1, image_height, device=dynamical_voxels_world_coo.device).view(1, image_height, 1).repeat(self.batch_size, 1, 1)
        u = u * (max_x - min_x).view(self.batch_size, 1, 1) + min_x.view(self.batch_size, 1, 1)
        v = v * (max_y - min_y).view(self.batch_size, 1, 1) + min_y.view(self.batch_size, 1, 1)
        u = u.expand(-1, image_width, -1)
        v = v.expand(-1, -1, image_height)
        uv_grid = torch.stack((u, v), dim=-1)
        uv_grid = uv_grid.view(self.batch_size, image_width * image_height, 2)

        uv_grid, cam_loc = get_rays(uv_grid, self.intrinsics, self.pose)   # This has to be changed with intrinsics and pose if cameras were normalized
        uv_grid = uv_grid.view(self.batch_size, image_width, image_height, 1, 3).repeat(1, 1, 1, depth_dep_res, 1)

        depth_values = torch.norm(dynamical_voxels_world_coo - self.cam_loc.unsqueeze(1), dim=-1)
        z_min, z_max = depth_values.min(dim=1)[0], depth_values.max(dim=1)[0]  
        #z_min, z_max = z_min*0 + 0.5, z_max*0 + 3.5
        z_vals = torch.linspace(0, 1, depth_dep_res, device=dynamical_voxels_world_coo.device).view(1, 1, 1, depth_dep_res).repeat(self.batch_size, image_width, image_height, 1)
        z_vals = z_min.view(self.batch_size, 1, 1, 1) + z_vals * (z_max - z_min).view(self.batch_size, 1, 1, 1)

        uvz_coo = cam_loc.view(self.batch_size, 1, 1, 1, 3) + uv_grid * z_vals.unsqueeze(-1).repeat(1, 1, 1, 1, 3)
        uvz_coo = uvz_coo.view(self.batch_size, -1, 3)
        uvz_canonical, out_mask = self.canonical_tfs_from_dynamical(uvz_coo, self.smpl_tfs, self.betas, self.gender, self.smpl_outputs, batch=True)
        #uvz_canonical = uvz_canonical * (1 - out_mask.float().unsqueeze(-1))
        #mask_cano_coo = (uvz_canonical.sum(dim=-1) != 0).unsqueeze(-1)
        mask_cano_coo = (1 - out_mask.float().unsqueeze(-1))
        #uvz_canonical = uvz_canonical - (out_mask.float().unsqueeze(-1))
        uvz_canonical = uvz_canonical.view(self.batch_size, image_width, image_height, self.opt['decoder']['depth_dep_res'], 3)
        mask_cano_coo = mask_cano_coo.view(self.batch_size, image_width, image_height, self.opt['decoder']['depth_dep_res'], 1)
        #uvz_canonical = uvz_canonical.permute(0, 4, 3, 2, 1)
        #mask_cano_coo = mask_cano_coo.permute(0, 4, 3, 2, 1)
        uvz_canonical = uvz_canonical.permute(0, 4, 3, 1, 2)
        mask_cano_coo = mask_cano_coo.permute(0, 4, 3, 1, 2)

        return uvz_canonical, mask_cano_coo

    def interpolate_to_voxels(self, features, voxels_coo, voxels_uv, rgb=False):
        '''This method is used to interpolate the features to the voxels.'''
        if not rgb:
            if features.shape[2] == self.img_size[0] and features.shape[3] == self.img_size[1]:   # pixel-aligned features (unet encoder)
                x_int = voxels_uv[..., 0].long()
                y_int = voxels_uv[..., 1].long()
                x_frac = voxels_uv[..., 0] - x_int.float()
                y_frac = voxels_uv[..., 1] - y_int.float()
            else:
                x_int = ((voxels_uv[..., 0] / self.img_size[1]) * features.shape[3]).long()
                y_int = ((voxels_uv[..., 1] / self.img_size[0]) * features.shape[2]).long()
                x_frac = (voxels_uv[..., 0] / self.img_size[1] * features.shape[3] - x_int.float())
                y_frac = (voxels_uv[..., 1] / self.img_size[0] * features.shape[2] - y_int.float())
            
            # Use advanced indexing to gather features 
            batch_indices = torch.arange(features.shape[0]).unsqueeze(1)
            #interpolated_features = torch.zeros(self.batch_size, voxels_coo.shape[1], features.shape[1]).to(features.device)
            #interpolated_features[..., :] = features[torch.arange(features.shape[0]).unsqueeze(1), :, y_int, x_int]
            x_frac = x_frac.unsqueeze(-1).repeat(1, 1, features.shape[1])
            y_frac = y_frac.unsqueeze(-1).repeat(1, 1, features.shape[1])
            interpolated_features = features[batch_indices, :, y_int, x_int] * x_frac * y_frac + \
                                            features[batch_indices, :, y_int, torch.clamp(x_int+1, 0, features.shape[3] - 1)] * (1 - x_frac) * y_frac + \
                                            features[batch_indices, :, torch.clamp(y_int+1, 0, features.shape[2] - 1), x_int] * x_frac * (1 - y_frac) + \
                                            features[batch_indices, :, torch.clamp(y_int+1, 0, features.shape[2] - 1), torch.clamp(x_int+1, 0, features.shape[3] - 1)] * (1 - x_frac) * (1 - y_frac)
            if False:   # Use non-relative depth values, aka fixed depth relative position
                depth_values = torch.norm(voxels_coo - self.cam_loc.unsqueeze(1), dim=-1)
                center_depth = depth_values.mean(dim=1)
                d_min = center_depth - 0.75
                d_max = center_depth + 0.75
                depth_values = (depth_values - d_min.view(-1, 1)) / (d_max.view(-1, 1) - d_min.view(-1, 1))        
            elif False:   # Consistent depth description
                depth_values = torch.norm(voxels_coo - self.cam_loc.unsqueeze(1), dim=-1)
                d_min = 0.5
                d_max = 3.5
                depth_values = (depth_values - d_min) / (d_max - d_min)
            else:
                depth_values = torch.norm(voxels_coo - self.cam_loc.unsqueeze(1), dim=-1)
                depth_values = (depth_values - depth_values.min(dim=1, keepdim=True)[0]) / (depth_values.max(dim=1, keepdim=True)[0] - depth_values.min(dim=1, keepdim=True)[0] + 1e-6) 
            z_int = (depth_values * (features.shape[1] - 1)).long()
            z_frac = (depth_values * (features.shape[1] - 1) - z_int.float())
            #features = interpolated_features[torch.arange(interpolated_features.shape[0]).unsqueeze(1), torch.arange(interpolated_features.shape[1]), z_int]
            voxels_indices = torch.arange(interpolated_features.shape[1])
            features = interpolated_features[batch_indices, voxels_indices, z_int] * z_frac + \
                        interpolated_features[batch_indices, voxels_indices, torch.clamp(z_int+1, max=features.shape[1]-1)] * (1 - z_frac)
            features = features.unsqueeze(-1)
        if rgb:   # (batch_size, 3, D, H, W)
            H, W = features.shape[3], features.shape[4]
            if H == self.img_size[0] and W == self.img_size[1]:   # pixel-aligned features (unet encoder)
                x_int = voxels_uv[..., 0].long()
                y_int = voxels_uv[..., 1].long()
                x_frac = voxels_uv[..., 0] - x_int.float()
                y_frac = voxels_uv[..., 1] - y_int.float()
            else:
                x_int = ((voxels_uv[..., 0] / self.img_size[1]) * W).long()
                y_int = ((voxels_uv[..., 1] / self.img_size[0]) * H).long()
                x_frac = (voxels_uv[..., 0] / self.img_size[1] * W - x_int.float())
                y_frac = (voxels_uv[..., 1] / self.img_size[0] * H - y_int.float())
            
            # Use advanced indexing to gather features 
            batch_indices = torch.arange(features.shape[0]).unsqueeze(1)
            #interpolated_features = torch.zeros(self.batch_size, voxels_coo.shape[1], features.shape[1]).to(features.device)
            #interpolated_features[..., :] = features[torch.arange(features.shape[0]).unsqueeze(1), :, y_int, x_int]
            x_frac = x_frac.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, features.shape[1], features.shape[2])
            y_frac = y_frac.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, features.shape[1], features.shape[2])
            interpolated_features = features[batch_indices, :, :, y_int, x_int] * x_frac * y_frac + \
                                            features[batch_indices, :, :, y_int, torch.clamp(x_int+1, 0, W - 1)] * (1 - x_frac) * y_frac + \
                                            features[batch_indices, :, :, torch.clamp(y_int+1, 0, H - 1), x_int] * x_frac * (1 - y_frac) + \
                                            features[batch_indices, :, :, torch.clamp(y_int+1, 0, H - 1), torch.clamp(x_int+1, 0, W - 1)] * (1 - x_frac) * (1 - y_frac)
            if False:   # Use non-relative depth values, aka fixed depth relative position
                depth_values = torch.norm(voxels_coo - self.cam_loc.unsqueeze(1), dim=-1)
                center_depth = depth_values.mean(dim=1)
                d_min = center_depth - 0.75
                d_max = center_depth + 0.75
                depth_values = (depth_values - d_min.view(-1, 1)) / (d_max.view(-1, 1) - d_min.view(-1, 1))
            elif False:   # Consistent depth description
                depth_values = torch.norm(voxels_coo - self.cam_loc.unsqueeze(1), dim=-1)
                d_min = 0.5
                d_max = 3.5
                depth_values = (depth_values - d_min) / (d_max - d_min)
            else:
                depth_values = torch.norm(voxels_coo - self.cam_loc.unsqueeze(1), dim=-1)
                depth_values = (depth_values - depth_values.min(dim=1, keepdim=True)[0]) / (depth_values.max(dim=1, keepdim=True)[0] - depth_values.min(dim=1, keepdim=True)[0] + 1e-6) 
            D = features.shape[2]
            z_int = (depth_values * (D - 1)).long()
            z_frac = (depth_values * (D - 1) - z_int.float())
            z_frac = z_frac.unsqueeze(-1).repeat(1, 1, features.shape[1])
            
            voxels_indices = torch.arange(interpolated_features.shape[1])
            features = interpolated_features[batch_indices, voxels_indices, :, z_int] * z_frac + \
                        interpolated_features[batch_indices, voxels_indices, :, torch.clamp(z_int+1, max=D-1)] * (1 - z_frac)

        return features
    
    def voxel_mapping(self, matrix_mapping, voxels_coo=None, voxels_values=None):
        dim_min = 0
        dim_max = self.mapping_dim - 1

        if voxels_coo is None:   # Returns the active voxels coo from a full matrix mapping
            active_voxels_coo = torch.nonzero(matrix_mapping == 1).float()
            active_voxels_coo = 2 * (active_voxels_coo - dim_min) / (dim_max - dim_min) - 1  
            return active_voxels_coo
        else:   # Returns the matrix mapping corresponding to the active voxels coo (voxels coo should be the cano values)
            reconstructed_matrix = torch.zeros_like(matrix_mapping).to(matrix_mapping.device)
            voxels_coo = ((voxels_coo + 1) / 2 * (dim_max - dim_min)) + dim_min 
            voxels_coo = voxels_coo.round().long()

            if voxels_values is None:   # Adds values to the positions corresponding to the active voxels coo
                reconstructed_matrix[voxels_coo[:, 0], voxels_coo[:, 1], voxels_coo[:, 2]] = 1
            else:   # This does not handle the batch
                assert voxels_values.dim() == 1
                assert voxels_values.shape[0] == voxels_coo.shape[0]
                reconstructed_matrix[voxels_coo[:, 0], voxels_coo[:, 1], voxels_coo[:, 2]] = voxels_values.squeeze(-1)

            return reconstructed_matrix

    def dynamical_tfs_from_canonical(self, points_coo, smpl_tfs, betas, gender, smpl_outputs, batch=False):
        if batch:
            x_d = torch.zeros(self.batch_size, points_coo.shape[1], points_coo.shape[2]).cuda().float()
            for i in range(self.batch_size):
                #smpl_verts = self.smpl_verts_cano * 1.1
                smpl_verts = self.smpl_verts_cano   # TODO: fix to handle different frames from different sequences
                deformer = SMPLDeformer(betas=betas[i], gender=gender[i])
                #x_d[i], outlier_mask = deformer.forward(points_coo[i], smpl_tfs[i], return_weights=False, inverse=False, smpl_verts=smpl_verts)
                x_d[i], outlier_mask = deformer.forward(points_coo[i], smpl_tfs[i], return_weights=False, inverse=False, smpl_verts=smpl_verts)
            if False:
                if self.visualize_stats:
                    with torch.no_grad():
                        self.visualize_voxels(points_coo[0][~outlier_mask], output_file='canonical_non_outliers.png', world=False)
                        self.visualize_voxels(x_d[0][~outlier_mask], output_file='dynamical_non_outliers.png', world=True)
        return x_d

    def canonical_tfs_from_dynamical(self, points_coo, smpl_tfs, betas, gender, smpl_outputs, batch=False):
        if batch:
            x_c = torch.zeros(self.batch_size, points_coo.shape[1], points_coo.shape[2]).cuda().float()
            outlier_mask = torch.zeros(self.batch_size, points_coo.shape[1]).cuda().bool()
            smpl_verts = smpl_outputs['smpl_verts'] 
            #smpl_verts_cano = self.smpl_verts_cano 
            #x_d = torch.zeros(self.batch_size, points_coo.shape[1], points_coo.shape[2]).cuda().float()
            for i in range(self.batch_size):
                deformer = SMPLDeformer(betas=betas[i], gender=gender[i])
                x_c[i], outlier_mask[i] = deformer.forward(points_coo[i].clone(), smpl_tfs[i], return_weights=False, inverse=True, smpl_verts=smpl_verts[i])
                #x_d[i], outlier_mask = deformer.forward(x_c[i], smpl_tfs[i], return_weights=False, inverse=False, smpl_verts=smpl_verts_cano)
                #print("transformation average error: ", torch.norm(points_coo[i] - x_d[i]) / points_coo[i].shape[0])
            
            if False:
                with torch.no_grad():
                    self.visualize_voxels(points_coo[0], output_file='dynamical_after_transform.png', world=True)
                    self.visualize_voxels(points_coo[0][~outlier_mask], output_file='dynamical_non_outliers.png', world=True)
                    self.visualize_voxels(x_c[0][~outlier_mask], output_file='canonical_non_outliers.png', world=True)
                    self.visualize_voxels(x_c[0], output_file='canonical_after_transform.png', world=True)
                    self.visualize_voxels(x_d[0], output_file='dynamical_after_inverse.png', world=True)
        return x_c, outlier_mask
    
    def generate_mesh(self, voxels_coo, occupancy_field, voxels_rgb, outputs_folder, frame_id=None, mode='cano'):
        '''This method is used to generate a mesh from the voxels.
        '''
        with torch.no_grad():
            if mode == 'cano':   # Canonical shape from single frame predictions
                print("cano mesh generation: canonical mesh from single-frame interpolated occupancy values")
                for i in range(self.batch_size):
                    t0_mesh = time.time()
                    ov_mask = (occupancy_field[i] > self.occupancy_threshold).squeeze(-1).float()
                    active_voxels = self.voxel_mapping(self.matrix_mapping)
                    voxels_matrix = self.voxel_mapping(self.matrix_mapping, voxels_coo=active_voxels, voxels_values=ov_mask)
                    
                    verts, faces, normals, values = measure.marching_cubes(voxels_matrix.cpu().numpy())
                    
                    rgb_values = voxels_rgb[i][ov_mask > 0.5]
                    rgb_values = torch.cat((rgb_values, occupancy_field[i][ov_mask > 0.5]), dim=-1).cpu().numpy()
                    rgb_values = (rgb_values * 255).astype(np.uint8)
                    mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=normals, vertex_colors=values)
                    #mesh.fill_holes()

                    if frame_id is None:
                        output_path = 'outputs/mesh.ply'
                    else:
                        if type(frame_id) == torch.Tensor:
                            output_path = os.path.join(outputs_folder, f'mesh_{frame_id.item()}_{i}.ply')
                        elif type(frame_id) == int:
                            output_path = os.path.join(outputs_folder, f'mesh_{frame_id}_{i}.ply')
                        else: 
                            raise ValueError("frame_id format not supported")
                    mesh.export(output_path)
                    t1_mesh = time.time()
                    wandb.log({"Time for mesh generation": t1_mesh-t0_mesh})
            elif mode == 'ao_cano':   # Canonical shape from whole sequence retained representation
                print("ao_cano mesh generation: canonical mesh from activity occupancy values")
                assert voxels_coo is None, "voxels_coo should be None for active occupancy canonical reconstruction"
                t0_mesh = time.time()
                ov_mask = (occupancy_field > self.opt['ao_threshold']).squeeze(-1).float()
                active_voxels = self.voxel_mapping(self.matrix_mapping)   # Retrieve cano coords from matrix mapping
                voxels_matrix = self.voxel_mapping(self.matrix_mapping, active_voxels, ov_mask)
                #voxels_matrix = upsample_matrix(voxels_matrix, scale_factor=2, threshold=0.5)
                
                verts, faces, normals, values = measure.marching_cubes(voxels_matrix.cpu().numpy())
                
                rgb_values = voxels_rgb[ov_mask.bool()]
                rgb_values = torch.cat((rgb_values, occupancy_field[ov_mask.bool()]), dim=-1).cpu().numpy()
                rgb_values = (rgb_values * 255).astype(np.uint8)

                #if verts.shape[0] != rgb_values.shape[0]:
                #    raise ValueError(f"Number of vertices ({verts.shape[0]}) and RGB values ({rgb_values.shape[0]}) do not match")

                #mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=rgb_values)
                mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=normals, vertex_colors=values)
                #mesh.fill_holes()

                if frame_id is None:
                    output_path = 'outputs/mesh.ply'
                else:
                    output_path = os.path.join(outputs_folder, f'mesh_epoch_{frame_id}.ply')
                mesh.export(output_path)
                t1_mesh = time.time()
                wandb.log({"Time for mesh generation": t1_mesh-t0_mesh})
            elif mode == 'dynamical_pc':   # Point cloud reconstruction in the dynamical space
                print("dynamical_pc mesh generation: dynamical mesh from dynamical space occupancy_field values")
                t0_mesh = time.time()
                ov_mask = (occupancy_field > 0.5).float()

                depth_values = torch.norm(voxels_coo - self.cam_loc.unsqueeze(1), dim=-1)
                min_z, max_z = depth_values.min(dim=1)[0], depth_values.max(dim=1)[0]
                depth_range = max_z - min_z
                depth_scale = depth_range / self.opt['decoder']['depth_dep_res']

                occupancy_field = occupancy_field * ov_mask
                voxels_rgb = voxels_rgb * ov_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
                
                for i in range(self.batch_size):
                    voxels_matrix = occupancy_field[i]
                    #voxels_matrix = voxels_matrix.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, 32, 384, 384)
                    #voxels_matrix = F.interpolate(voxels_matrix, size=(128, 768, 768), mode='trilinear', align_corners=True)
                    #voxels_matrix = voxels_matrix.squeeze(0).squeeze(0)
                    rgb_values = voxels_rgb[i]
                    #rgb_values = rgb_values.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, 32, 384, 384)
                    #rgb_values = F.interpolate(rgb_values, size=(384, 384, 384), mode='trilinear', align_corners=True)
                    #rgb_values = rgb_values.squeeze(0).squeeze(0)
                
                    verts, faces, normals, values = measure.marching_cubes(voxels_matrix.cpu().numpy(), level=0.5)
                
                    mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=normals, vertex_colors=values)
                    #mesh.fill_holes()

                if frame_id is None:
                    output_path = 'outputs/mesh.ply'
                else:
                    output_path = os.path.join(outputs_folder, f'dynamical_epoch_{frame_id}.ply')
                mesh.export(output_path)
                t1_mesh = time.time()
                wandb.log({"Time for mesh generation": t1_mesh-t0_mesh})
            elif mode == 'dynamical_ao':   # Canonical shape from whole sequence retained representation
                print("dynamical_ao mesh generation: dynamical mesh from activity occupancy values")
                t0_mesh = time.time()   
                ov_mask = (occupancy_field > self.opt['ao_threshold']).squeeze(-1).float()
                active_voxels = self.voxel_mapping(self.matrix_mapping)   # Retrieve cano coords from matrix mapping
                voxels_matrix = self.voxel_mapping(self.matrix_mapping, active_voxels, ov_mask)
                #voxels_matrix = upsample_matrix(voxels_matrix, scale_factor=2, threshold=0.5)

                verts, faces, normals, values = measure.marching_cubes(voxels_matrix.cpu().numpy())
                
                rgb_values = voxels_rgb[ov_mask.bool()]
                rgb_values = torch.cat((rgb_values, occupancy_field[ov_mask.bool()]), dim=-1).cpu().numpy()
                rgb_values = (rgb_values * 255).astype(np.uint8)

                #if verts.shape[0] != rgb_values.shape[0]:
                #    raise ValueError(f"Number of vertices ({verts.shape[0]}) and RGB values ({rgb_values.shape[0]}) do not match")

                for i in range(self.batch_size):
                    #mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=rgb_values)
                    mesh_canonical = trimesh.Trimesh(vertices=verts, faces=faces, normals=normals, vertex_colors=values)   # TODO: maybe process false
                    deformer = SMPLDeformer(betas=self.betas[i], gender=self.gender[i], K=7)
                    verts_i = verts.copy()
                    verts_i = torch.tensor(verts_i).cuda().float()
                    weights = deformer.query_weights(verts_i)
                    verts_deformed = skinning(verts_i.unsqueeze(0),  weights, self.smpl_tfs[i]).data.cpu().numpy()[0]
                    mesh = trimesh.Trimesh(vertices=verts_deformed, faces=mesh_canonical.faces, process=True)   
                    #mesh.fill_holes()

                    if frame_id is None:
                        output_path = 'outputs/mesh.ply'
                    else:
                        output_path = os.path.join(outputs_folder, f'mesh_epoch_{frame_id}.ply')
                    mesh.export(output_path)
                    t1_mesh = time.time()
                    wandb.log({"Time for mesh generation": t1_mesh-t0_mesh})

        return mesh
    
    def render_image(self, dynamical_voxels_coo, voxels_ov, voxels_rgb, original_image, depth_image):
        '''Render image from occupancy field and rgb field.'''
        with torch.no_grad():
            ray_caster = RayCaster(self, dynamical_voxels_coo, voxels_ov, voxels_rgb, original_image, depth_image)
            rendered_image = ray_caster.render_values_at_rays(mode='render_image')

        return rendered_image
    
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def visualize_voxels(self, voxels_coo, add_point=None, output_file=None, world=False, voxels_ov=None, voxels_rgb=None):
        '''This is used to save an image of the voxels in 3D space.
        '''
        with torch.no_grad():
            voxels_coo_np = voxels_coo.detach().cpu().numpy()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            if voxels_ov is not None and voxels_rgb is None:
                voxels_ov_np = voxels_ov.detach().cpu().numpy()
                ax.scatter(voxels_coo_np[:, 0], voxels_coo_np[:, 1], voxels_coo_np[:, 2], alpha=voxels_ov_np)
            elif voxels_rgb is not None and voxels_ov is None:
                voxels_rgb_np = voxels_rgb.detach().cpu().numpy()
                ax.scatter(voxels_coo_np[:, 0], voxels_coo_np[:, 1], voxels_coo_np[:, 2], c=voxels_rgb_np)
            if voxels_ov is not None and voxels_rgb is not None: 
                voxels_ov_np = voxels_ov.detach().cpu().numpy()
                voxels_rgb_np = voxels_rgb.detach().cpu().numpy()
                ax.scatter(voxels_coo_np[:, 0], voxels_coo_np[:, 1], voxels_coo_np[:, 2], c=voxels_rgb_np, alpha=voxels_ov_np)
            else:
                ax.scatter(voxels_coo_np[:, 0], voxels_coo_np[:, 1], voxels_coo_np[:, 2], alpha=0.1)

            if add_point is not None:
                ax.scatter(add_point[:, 0], add_point[0, 1], add_point[0, 2], c='r', marker='o', s=100)

            ax.set_title(f'Voxel Visualization (occupied voxels: {voxels_coo_np.shape[0]})')

            if not world:
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_xlim([-1, 1])
                ax.set_ylim([-1, 1])
                ax.set_zlim([-1, 1])
            
            if output_file is None:
                output_file = 'voxels_temp.png'
            output_file = os.path.join('outputs/stats/', output_file)
            wandb.log({output_file: [wandb.Image(fig)]})
            plt.close('all')

        return