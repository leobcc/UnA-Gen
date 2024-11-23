import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import DoubleConv, DownSample, UpSample
from lib.deformer import SMPLDeformer, skinning
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from torchvision.utils import make_grid, save_image
from torchvision import transforms, models
import time
from lib.utils import get_camera_loc, coordinates_projection_map, get_cam, get_rays, get_uv, upsample_matrix, quat_to_rot, axis_angle_to_rotation_matrix, get_global_transformation
from lib.utils import stable_softmax
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
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
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//2, out_channels=in_features//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//2, out_channels=in_features//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//2, out_channels=in_features//2, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=1),
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
            nn.Conv2d(in_channels=input_dim, out_channels=(input_dim)//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//2, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
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
            nn.Conv2d(in_channels=input_dim, out_channels=(input_dim)//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//2, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
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
            nn.Conv2d(in_channels=input_dim, out_channels=(input_dim)//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//2, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
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
            nn.Conv2d(in_channels=input_dim, out_channels=(input_dim)//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//2, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
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
    
class OccupancyField_cnn_v2(nn.Module):
    def __init__(self, in_features=128, hidden_features=32, out_channels=1):
        super(OccupancyField_cnn_v2, self).__init__()

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
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=(input_dim)//2, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        input_dim = (input_dim)//2+hidden_features
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=(input_dim)//2, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        input_dim = (input_dim)//2+hidden_features
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),    
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
            nn.Conv2d(in_channels=input_dim, out_channels=(input_dim)//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//2, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=(input_dim)//2, out_channels=(input_dim)//4, kernel_size=3, stride=1, padding=1),
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
    
class RGBfield_cnn_v2(nn.Module):
    def __init__(self, in_features=128, hidden_features=32, out_channels=3):
        super(RGBfield_cnn_v2, self).__init__()

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
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=(input_dim)//2, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        input_dim = (input_dim)//2+hidden_features
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=(input_dim)//2, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        input_dim = (input_dim)//2+hidden_features
        self.conv_r = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),            
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),        
        )
        self.conv_b = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_dim, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),          
        )

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
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
        self.intrinsics = None   
        self.smpl_params = None
        self.pose = None
        self.cam_loc = None
        self.scale = None
        self.trans = None
        self.standard_depth_n = torch.nn.Parameter(torch.tensor(opt['standard_depth_n'], dtype=torch.float32))
        self.standard_depth_f = opt['standard_depth_f']
        self.n_training_rays = opt['n_training_rays']
        self.mapping_dim = opt['mapping_dim']
        self.occupancy_threshold = opt['occupancy_threshold']
        self.matrix_mapping = self.initialize_matrix_mapping(self.mapping_dim, 0.5).cuda()
        self.activity_occupancy = None
        self.activity_occupancy_rgb = None
        self.ao_count = None
        self.ao = None
        self.ao_rgb = None
        self.mapping_prob_density = torch.zeros_like(self.matrix_mapping).cuda()

        self.prev_of = None
        self.prev_rgb = None

        if opt['visualize_stats'] != 'epoch_end':
            self.visualize_stats = opt['visualize_stats']

        # Unet encoder initialization
        if opt['encoder']['name'] == 'unet':
            self.encoder = UNetEncoder(in_channels=in_channels, out_channels=features).cuda()
            if not opt['encoder']['train']:
                self.encoder.eval()
                for param in self.encoder.parameters():
                    param.requires_grad = False

        # DepthAnything encoder initialization
        if opt['encoder']['name'] == 'dpt':
            encoder = opt['encoder']['size']   
            pretrained_encoder = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder))
            print('DepthAnything model loaded as pretrained encoder')
            print('2 layers dropped from the output_conv2. Encoding dimension: 32x924x518')
            self.dptEncoder = dptEncoder(pretrained_encoder, layers_to_drop=2).cuda()   # This is the wrapper used to remove the last layers

            if not opt['encoder']['train']:
                self.dptEncoder.eval()
                for param in self.dptEncoder.parameters():
                    param.requires_grad = False
        
        # Resnet encoder initialization
        if opt['encoder']['name'] == 'resnet':
            self.ResNetEncoder = ResNetEncoder().cuda()
            if not opt['encoder']['train']:
                self.ResNetEncoder.eval()
                for param in self.ResNetEncoder.parameters():
                    param.requires_grad = False

        # Unet_resnet encoder initialization
        if opt['encoder']['name'] == 'unet_resnet':
            self.ResNetEncoder = smp.Unet('resnet34', classes=128, activation='softmax')
            if not opt['encoder']['train']:
                self.ResNetEncoder.eval()
                for param in self.ResNetEncoder.parameters():
                    param.requires_grad = False
        
        # CNN encoder initialization
        if opt['encoder']['name'] == 'cnn':
            self.CNNEncoder = CNNEncoder(in_channels=3, out_channels=features).cuda()
            if not opt['encoder']['train']:
                self.CNNEncoder.eval()
                for param in self.CNNEncoder.parameters():
                    param.requires_grad = False

        # Combine the encoders
        if opt['encoder']['name'] == 'comb':
            encoder = opt['encoder']['size']   
            pretrained_encoder = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder))
            print('DepthAnything model loaded as pretrained encoder')
            print('2 layers dropped from the output_conv2. Encoding dimension: 32x924x518')
            self.dptEncoder = dptEncoder(pretrained_encoder, layers_to_drop=2).cuda()   # This is the wrapper used to remove the last layers
            if not opt['encoder']['train']:
                self.dptEncoder.eval()
                for param in self.dptEncoder.parameters():
                    param.requires_grad = False
            
            self.ResNetEncoder = ResNetEncoder().cuda()
            if not opt['decoder']['train']:
                self.ResNetEncoder.eval()
                for param in self.ResNetEncoder.parameters():
                    param.requires_grad = False

        '''
        for param in self.GeometryEncoder.parameters():
            param.requires_grad = False

        for param in self.RenderingEncoder.parameters():
            param.requires_grad = False
        '''

        if self.opt['add_cano_coo'] and not self.opt['add_depth_features']:
            self.OccupancyField = OccupancyField_cnn_v2(in_features=features+4, out_channels=self.opt['decoder']['depth_dep_res']).cuda()
            self.RGBField = RGBfield_cnn_v2(in_features=features+4, out_channels=self.opt['decoder']['depth_dep_res']).cuda()      
            if self.opt['shadow_field']:      
                self.Shader = RGBfield_cnn(in_features=features, out_channels=self.opt['decoder']['depth_dep_res']*2).cuda()
        elif self.opt['add_depth_features'] and not self.opt['add_cano_coo']:
            self.OccupancyField = OccupancyField_cnn(in_features=features+1, out_channels=self.opt['decoder']['depth_dep_res']).cuda()
            self.RGBField = RGBfield_cnn(in_features=features+1, out_channels=self.opt['decoder']['depth_dep_res']).cuda()   
            if self.opt['shadow_field']:           
                self.Shader = RGBfield_cnn(in_features=features, out_channels=self.opt['decoder']['depth_dep_res']*2).cuda()
        elif self.opt['add_cano_coo'] and self.opt['add_depth_features']:
            self.OccupancyField = OccupancyField_cnn_v2(in_features=features+5, out_channels=self.opt['decoder']['depth_dep_res']).cuda()
            self.RGBField = RGBfield_cnn_v2(in_features=features+5, out_channels=self.opt['decoder']['depth_dep_res']).cuda()   
            if self.opt['shadow_field']:           
                self.Shader = RGBfield_cnn(in_features=features, out_channels=self.opt['decoder']['depth_dep_res']*2).cuda()
        else:
            self.OccupancyField = OccupancyField_cnn(in_features=features, out_channels=self.opt['decoder']['depth_dep_res']).cuda()
            self.RGBField = RGBfield_cnn(in_features=features, out_channels=self.opt['decoder']['depth_dep_res']).cuda()
            if self.opt['shadow_field']:  
                self.Shader = RGBfield_cnn(in_features=features, out_channels=self.opt['decoder']['depth_dep_res']*2).cuda()

        #self.shadow_field = None   # TODO: remove this, used oonly for visualization
        self.depth_image = None   # TODO: remove this, used only for visualization

        self.apply(self.weights_init)   # weights initialization

    def forward(self, inputs):
        # input is a dictionary containing the image, the smpl parameters, and other info 
        # it is of shape (batch_size, key_values) where key values is of different dimensions
        t0 = time.time()  
        image = inputs['masked_image']
        #image_dpt = inputs['image_dpt'].cuda()
        image_resnet = inputs['image_resnet'].cuda()
        #image_unet_resnet = inputs['image_unet_resnet'].cuda()
        self.img_size = (image.shape[2], image.shape[3])
        self.batch_size = image.shape[0]
        # resize_transform = transforms.Resize((512, 512))
        # image = resize_transform(original_image)
        self.original_size = (inputs['original_size'][0].cuda(), inputs['original_size'][1].cuda())
        if self.batch_size == 1:
            self.image_size = torch.tensor([image.shape[-2], image.shape[-1]]).cuda()
        else:
            self.image_size = torch.stack(inputs['image_size']).cuda()
        #assert self.image_size == image.shape[-2:], "Image size mismatch"
        self.min_x = inputs['min_x'].cuda()
        self.max_x = inputs['max_x'].cuda()   # TODO: keep only what needed and change way of storing
        self.min_y = inputs['min_y'].cuda()
        self.max_y = inputs['max_y'].cuda()
        batch_idx = inputs['batch_idx']
        num_samples = inputs['num_samples']
        if self.opt['visualize_stats'] == 'epoch_end':
            if batch_idx == num_samples - 1:
                self.visualize_stats = True
                wandb.log({"epoch": inputs['epoch']})
            else:
                self.visualize_stats = False
        depth_image = inputs['depth_image'].squeeze(1).cuda()

        self.depth_image = depth_image   # TODO: remove this, used only for visualization

        gender = inputs['metadata']['gender']
        betas = inputs['betas']
        # scale = inputs['smpl_params'][:, 0].cuda()
        self.pose = inputs["pose"].cuda()
        self.intrinsics = inputs['intrinsics'].cuda()
        self.smpl_params = inputs['smpl_params'].cuda()
        self.smpl_verts_cano = inputs['smpl_verts_cano'].cuda()
        smpl_tfs = inputs['smpl_tfs'].cuda()
        smpl_outputs = inputs['smpl_outputs']
        self.scale = torch.tensor(1.1).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, 3).cuda()
        self.trans = torch.tensor(0.0).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, 3).cuda()

        with torch.no_grad():
            # Matrix mapping refinement
            if self.opt['active_occupancy_refinement'] and (inputs['epoch']+1) % self.opt['active_occupancy_refinement_epochs'] == 0 and batch_idx == 0:
                print("Active occupancy refinement at epoch", inputs['epoch'])
                self.active_occupancy_refinement(mode='preserve')
                self.opt['correct'] = True
                #confs['loss']['canonical_consistency_loss'] = True

            matrix_mapping = self.matrix_mapping

        
            # Learn normalization transformation for the metric spaces
            # I <> D <> C   I: Image space, D: Dynamical space (metric), C: Canonical space (metric)
            #      |    |
            #      D    C   D: Dynamical space (normalized), C: Canonical space (normalized)
            # The metric spaces are used to establish the 1-1 correspondence by skinning
            # The normalized spaces are used to establish the 1-1 correspondence by the feature encoder and evaluate the networks
            self.cam_loc = get_cam(self.pose)   # These parameters are used throught many methods

            # The active voxels are mapped by the matrix_mapping (while in the canonical space)
            # TODO: Implement the mapping of the active voxels by the matrix_mapping 
            active_voxels = self.voxel_mapping(matrix_mapping)   # Returns the active voxels coordinates in the canonical space
            if self.visualize_stats:
                self.visualize_voxels(active_voxels, output_file='active_voxels.png')

            # The active voxels are transformed to the dynamical space from the canonical space using the smpl pose parameters
            # TODO: Implement the transformation of the active voxels to the dynamical space using the smpl pose parameters
            # dynamical_voxels_coo = transform_voxels(active_voxels_coo, smpl_tfs)   # Returns the active voxels coordinates in the dynamical space

            canonical_voxels_coo = active_voxels.unsqueeze(0).repeat(self.batch_size, 1, 1) 
            #if self.training and not self.visualize_stats:   # During training we only select a few voxels to reduce batch memory a processing time
            #    n_voxels = canonical_voxels_coo.shape[1] * (inputs['epoch']+1) // 1000
            #    random_indices = torch.randint(0, canonical_voxels_coo.shape[1], (canonical_voxels_coo.shape[0], n_voxels), device=canonical_voxels_coo.device)
            #    canonical_voxels_coo = canonical_voxels_coo[torch.arange(self.batch_size).unsqueeze(1), random_indices]
            dynamical_voxels_world_coo = self.dynamical_tfs_from_canonical(canonical_voxels_coo.clone(), smpl_tfs, betas, gender, smpl_outputs, batch=True)

            if self.visualize_stats:
                try:
                    self.visualize_voxels(canonical_voxels_coo[0].clone(), output_file='world_canonical_voxels_coo.png', world=True)
                except:
                    pass
            
            voxels_uv = get_uv(dynamical_voxels_world_coo, self.intrinsics, self.pose)
            voxels_uv[..., 0] = (voxels_uv[..., 0] - self.min_x[:, None]) / (self.max_x[:, None] - self.min_x[:, None]) * torch.tensor(self.img_size[1], device=voxels_uv.device).repeat(self.batch_size)[:, None]
            voxels_uv[..., 1] = (voxels_uv[..., 1] - self.min_y[:, None]) / (self.max_y[:, None] - self.min_y[:, None]) * torch.tensor(self.img_size[0], device=voxels_uv.device).repeat(self.batch_size)[:, None]
            voxels_uv[..., 0] = torch.clamp(voxels_uv[..., 0], 0, self.img_size[1]-1)   # This supposes images have the same size, but it is a given that it is the case
            voxels_uv[..., 1] = torch.clamp(voxels_uv[..., 1], 0, self.img_size[0]-1)
            #voxels_uv[..., 0] = (voxels_uv[..., 0] - self.min_x) / self.original_size[1] * self.img_size[1] 
            #voxels_uv[..., 1] = (voxels_uv[..., 1] - self.min_y) / self.original_size[0] * self.img_size[0] 
            
            if self.visualize_stats:
               self.visualize_voxels(dynamical_voxels_world_coo[0].clone(), output_file='dynamical_voxels_world_coo.png', world=True)

        # ---
        # Trained unet encoder
        # The feature encoder produces a pixel/voxel-wise feature embedding used to compute the occupancy field of the active voxels
        if self.opt['encoder']['train']:   # The encoder is trained 
            if self.opt['encoder']['name'] == 'unet':
                    features = self.encoder(image)

            if self.opt['encoder']['name'] == 'dpt':
                #features = torch.zeros((self.batch_size, 32, 924, 518)).cuda()
                features = torch.zeros((self.batch_size, 128, 924, 518)).cuda()   # Change the number of features depending on the encoder size
                features = self.dptEncoder(image_dpt)

            if self.opt['encoder']['name'] == 'resnet':
                features = self.ResNetEncoder(image_resnet)

            if self.opt['encoder']['name'] == 'unet_resnet':
                features = self.ResNetEncoder(image_unet_resnet)

            if self.opt['encoder']['name'] == 'cnn':
                features = self.CNNEncoder(image)

        elif not self.opt['encoder']['train']:   # Training of the encoder is paused
            with torch.no_grad():   
                if self.opt['encoder']['name'] == 'unet':
                    features = self.encoder(image)

                if self.opt['encoder']['name'] == 'dpt':
                    #features = torch.zeros((self.batch_size, 32, 924, 518)).cuda()
                    features = torch.zeros((self.batch_size, 128, 924, 518)).cuda()   # Change the number of features depending on the encoder size
                    features = self.dptEncoder(image_dpt)

                if self.opt['encoder']['name'] == 'resnet':
                    features = self.ResNetEncoder(image_resnet)

                if self.opt['encoder']['name'] == 'unet_resnet':
                    features = self.ResNetEncoder(image_unet_resnet)

                if self.opt['encoder']['name'] == 'cnn':
                    features = self.CNNEncoder(image)
        
        # --- Add canonical coordinates to the features ---
        if self.opt['add_cano_coo']:
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

        if self.opt['add_depth_features']:
            features = torch.cat((features, depth_image.unsqueeze(1)), dim=1)

        if self.visualize_stats:
            with torch.no_grad():
                features_0 = features[0, :9].clone().detach().view(features.shape[2], -1, 3).cpu().numpy()
                wandb.log({'features': [wandb.Image(features_0)]})

        # --- Occupancy ---
        occupancy_field_t = self.OccupancyField(features)
        #of_grad_t0 = time.time()
        #occupancy_field_grad = torch.autograd.grad(outputs=occupancy_field_t, inputs=features,
        #                            grad_outputs=torch.ones_like(occupancy_field_t, requires_grad=False, device=occupancy_field_t.device),
        #                            create_graph=True, retain_graph=True, only_inputs=True)[0][:, :4]
        #of_grad_t1 = time.time()
        
        #weights = torch.linspace(1, 0, steps=occupancy_field_t.shape[1]).to(occupancy_field_t.device)
        #weights /= weights.sum()  
        #weights = weights.view(1, occupancy_field_t.shape[1], 1, 1)

        #of_dpt = (occupancy_field_t * weights).sum(dim=1)
        #min_val, max_val = torch.amin(of_dpt, dim=(-1, -2)), torch.amax(of_dpt, dim=(-1, -2))
        #of_dpt = (of_dpt - min_val.view(-1, 1, 1)) / (max_val - min_val).view(-1, 1, 1)
        #of_dpt = F.pad(of_dpt, (1, 1, 1, 1), value=0)

        outliers_mask = (1 - (image.mean(dim=1) > 0).float())

        if self.visualize_stats:
            with torch.no_grad():
                selected_images = occupancy_field_t[0, :6]  # This is now of shape [6, H, W]
                concatenated_images_0 = torch.cat(tuple(selected_images), dim=-1)
                selected_images = occupancy_field_t[0, 6:12]  # This is now of shape [6, H, W]
                concatenated_images_1 = torch.cat(tuple(selected_images), dim=-1)
                selected_images = occupancy_field_t[0, 12:18]  # This is now of shape [6, H, W]
                concatenated_images_2 = torch.cat(tuple(selected_images), dim=-1)
                selected_images = occupancy_field_t[0, -18:-12]  # This is now of shape [6, H, W]
                concatenated_images_3 = torch.cat(tuple(selected_images), dim=-1)
                selected_images = occupancy_field_t[0, -12:-6]  # This is now of shape [6, H, W]
                concatenated_images_4 = torch.cat(tuple(selected_images), dim=-1)
                selected_images = occupancy_field_t[0, -6:]  # This is now of shape [6, H, W]
                concatenated_images_5 = torch.cat(tuple(selected_images), dim=-1)
                concatenated_images = torch.cat((concatenated_images_0, concatenated_images_1, concatenated_images_2, concatenated_images_3,
                                                 concatenated_images_4, concatenated_images_5), dim=-2)
                concatenated_images_np = concatenated_images.cpu().numpy()
                wandb.log({"Occupancy field": [wandb.Image(concatenated_images_np, mode='L')]})

        
        #weights /= (weights).sum()  
        
        #weights = torch.exp(-torch.linspace(0, 1, steps=occupancy_field_t.shape[1], device=occupancy_field_t.device))
        #weights /= (weights).sum()
        #weights = weights.view(1, -1, 1, 1).expand_as(occupancy_field_t)

        weights = torch.linspace(1, 0, steps=occupancy_field_t.shape[1], device=occupancy_field_t.device)
        weights = weights.view(1, -1, 1, 1)
        cum_of = torch.clamp(torch.cumsum(occupancy_field_t, dim=1) - occupancy_field_t, 0, 1)
        of_dpt = (occupancy_field_t * weights * (1 - cum_of)).sum(dim=1)   

        #min_val, max_val = torch.amin(of_dpt, dim=(-1, -2)), torch.amax(of_dpt, dim=(-1, -2))
        #of_dpt = (of_dpt - min_val.view(-1, 1, 1)) / (max_val - min_val).view(-1, 1, 1)
        #of_dpt = of_dpt / (max_val.view(-1, 1, 1) + 1e-6)
        #of_dpt = F.pad(of_dpt, (1, 1, 1, 1), value=0)
        
        #cum_of_s = self.interpolate_to_voxels(cum_of, dynamical_voxels_world_coo, voxels_uv)
        of_diff = torch.abs(occupancy_field_t[:, 1:] - occupancy_field_t[:, :-1])
        #threshold = 0.1
        #app_mask = (of_diff > threshold).float()
        #of_diff = app_mask * of_diff * 10

        #cum_of = torch.cumsum(occupancy_field_t, dim=1) - occupancy_field_t
        #inv_cum_of = 1 / (cum_of + 1e-6)
        
        #cum_of = F.pad(cum_of, (1, 1, 1, 1), value=0)
        #double_cum_of = occupancy_field_t 
        #softmin = (occupancy_field_t>0.5).float() * (1-(cum_of*0.9))
        #softmin = softmin / (softmin.sum(dim=1, keepdim=True) + 1e-6)
        
        #occupancy_mask = (occupancy_field_t > self.occupancy_threshold).float()

        cum_of = torch.cumsum(occupancy_field_t, dim=1) - occupancy_field_t
        logits = (-(cum_of * occupancy_field_t) - (1 - occupancy_field_t)*occupancy_field_t.shape[1])*1000
        softmin = stable_softmax(logits, dim=1)
        front_voxels = softmin * occupancy_field_t
        
        #softmin = torch.softmax(-(cum_of * occupancy_field_t) - (1 - occupancy_field_t), dim=1)
        #softmin = torch.softmax(-(cum_of * occupancy_field_t) - (1 - occupancy_field_t)*occupancy_field_t.shape[1], dim=1)
        #front_voxels = softmin * torch.ones_like(occupancy_field_t)
        #front_voxels = softmin * occupancy_field_t
        ray_opacity = front_voxels.sum(dim=1)

        #of_dpt = (occupancy_field_t * weights).sum(dim=1)
        
        #softmin = torch.softmax(F.pad(double_cum_of * (1-cum_of), (1, 1, 1, 1), value=0), dim=1)
        #voxel_idxs = torch.linspace(1, 0, steps=softmin.shape[1], device=softmin.device).view(1, -1, 1, 1).repeat(self.batch_size, 1, softmin.shape[2], softmin.shape[3])
        #of_dpt = (voxel_idxs * softmin).sum(dim=1)
        #min_val, max_val = torch.amin(of_dpt, dim=(-1, -2)), torch.amax(of_dpt, dim=(-1, -2))
        #of_dpt = of_dpt / (max_val.view(-1, 1, 1) + 1e-6)

        #front_idxs = (1-cum_of).sum(dim=1)
        #front_idxs = (front_idxs * softmin).sum(dim=1) 
        #front_idxs = (front_idxs * softmin).sum(dim=1) * (image.mean(dim=1) > 0).float()
        #front_idxs = torch.argmin(F.pad(double_cum_of, (1, 1, 1, 1), value=0), dim=1) * (image.mean(dim=1) > 0).float()
        #d_max = torch.amax(front_idxs, dim=(-1, -2))
        #d_min = torch.amin(front_idxs + (1 - (image.mean(dim=1) > 0).float())*occupancy_field_t.shape[1], dim=(-1, -2))
        #of_dpt = torch.clamp(1 - (front_idxs.float() - d_min.view(-1, 1, 1)) / (d_max - d_min + 1e-6).view(-1, 1, 1), 0, 1) * (image.mean(dim=1) > 0).float()
        #of_dpt = (1 - front_idxs.float() / d_max.view(-1, 1, 1)) * (image.mean(dim=1) > 0).float()
        #of_dpt = (1 - (front_idxs.float() / occupancy_field_t.shape[1])) * (image.mean(dim=1) > 0).float()
        #of_dpt = (1 - (front_idxs.float() / occupancy_field_t.shape[1])) 
        stacked_images = torch.cat((of_dpt[0], depth_image[0]), dim=1)
        wandb.log({'of_dpt': [wandb.Image(stacked_images.clone().detach().cpu().numpy(), mode='L')]})

        '''
        front_idxs_tensor = torch.zeros_like(double_cum_of)
        front_idxs_tensor[torch.arange(self.batch_size), front_idxs.long()] = 1
        '''
        #front_idxs = ((occupancy_field_t * (1 - cum_of)) > 0).float()
        #front_idxs = occupancy_field_t * (1 - cum_of)
        occupancy_field = self.interpolate_to_voxels(occupancy_field_t, dynamical_voxels_world_coo, voxels_uv)
        softmin = self.interpolate_to_voxels(softmin, dynamical_voxels_world_coo, voxels_uv)
        activity_mask = (softmin > 0.1).float()    
        if self.visualize_stats: 
            wandb.log({'softmin': wandb.Histogram(softmin[0].view(-1).clone().detach().cpu().numpy())})
        '''
        front_idxs_tensor = self.interpolate_to_voxels(front_idxs_tensor, dynamical_voxels_world_coo, voxels_uv)
        activity_mask = (front_idxs_tensor > 0).float()
        '''
        batch_size, n_voxels, _ = occupancy_field.shape
        n_select = 1024
        indices = torch.randint(0, n_voxels, (batch_size, n_select))

        #indices = indices.unsqueeze(-1).expand(-1, -1, occupancy_field.shape[-1])
        #batch_indices = torch.arange(batch_size).unsqueeze(1).expand_as(indices[:,:,0])
        #selected_voxels = occupancy_field[batch_indices, indices[:,:,0], :]
        #occupancy_field_grad = torch.autograd.grad(outputs=selected_voxels, inputs=features,
        #                            grad_outputs=torch.ones_like(selected_voxels, requires_grad=False, device=selected_voxels.device),
        #                            create_graph=True, retain_graph=True, only_inputs=True)[0][:, :3]

        #occupancy_field_grad = torch.autograd.grad(outputs=occupancy_field, inputs=features,
        #                            grad_outputs=torch.ones_like(occupancy_field, requires_grad=False, device=occupancy_field.device),
        #                            create_graph=True, retain_graph=True, only_inputs=True)[0][:, :3]

        if self.visualize_stats:
            with torch.no_grad():
                occupancy_field_0 = occupancy_field[0].view(-1).clone().detach().cpu().numpy()
                wandb.log({"Occupancy field values distribution": wandb.Histogram(occupancy_field_0)})

                if self.visualize_stats:
                    occupancy_map = (occupancy_field[0] > self.occupancy_threshold).squeeze(-1)
                    occupied_world_canonical_voxels_coo = canonical_voxels_coo[0][occupancy_map]
                    self.visualize_voxels(occupied_world_canonical_voxels_coo, output_file='occupied_world_canonical_voxels_coo.png', world=True)

        # --- Rendering ---
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
        wandb.log({'rasterized_image': [wandb.Image(rasterized_image[0].clone().detach().permute(1, 2, 0).cpu().numpy())]})

        if self.visualize_stats:
            with torch.no_grad():
                try:
                    occupancy_map = (occupancy_field > 0.5).squeeze(-1)
                    occ_cano_voxels_coo = canonical_voxels_coo[occupancy_map]
                    occ_cano_voxels_ov = occupancy_field[occupancy_map]
                    occ_cano_voxels_rgb = rgb_field[occupancy_map]
                    self.visualize_voxels(occ_cano_voxels_coo.clone(), output_file='cano_voxels_rgb.png', world=True, voxels_ov=occ_cano_voxels_ov, voxels_rgb=occ_cano_voxels_rgb)
                except:
                    pass
        #shadow_field = self.Shader(features[:, 4:])
        #shadow_field = self.interpolate_to_voxels(shadow_field, dynamical_voxels_world_coo, voxels_uv)
        
        #voxels_rgb = torch.clamp(rgb_field * shadow_field.expand_as(rgb_field), 0, 1)   # Apply shadowing to the rgb field
                                                                     # voxels_rgb is the final rgb field to be rendered
                                                                     # rgb_field is the actual rgb field without shadowing

        #self.shadow_field = shadow_field   # TODO: remove this, now used only for visualization, but it wastes memory
        #voxels_rgb = rgb_field 
        # ---
        if self.visualize_stats:
            if self.visualize_stats:
                with torch.no_grad():
                    rgb_field_0 = rgb_field[0].clone().detach().cpu().numpy()
                    wandb.log({"RGB field values distribution": wandb.Histogram(rgb_field_0)})

                    #shadow_field_0 = shadow_field[0].clone().detach().cpu().numpy()
                    #wandb.log({"Shadow field values distribution": wandb.Histogram(shadow_field_0)})

                    voxels_rgb_0 = voxels_rgb[0].clone().detach().cpu().numpy()
                    wandb.log({"Voxels RGB values distribution": wandb.Histogram(voxels_rgb_0)})
        
        # --- Render values for optimization ---
        t0_render_rgb_values = time.time()
        if self.opt['n_training_rays'] > 0:
            ray_caster = RayCaster(self, dynamical_voxels_world_coo, occupancy_field, cum_of_s, voxels_rgb, image, depth_image)
            training_values = ray_caster.render_values_at_rays(mode='training')
        t1_render_rgb_values = time.time()

        if self.visualize_stats:
            wandb.log({"Time for rendering rgb values": (t1_render_rgb_values - t0_render_rgb_values)})

        depth_all = torch.norm(dynamical_voxels_world_coo - self.cam_loc.unsqueeze(1), dim=-1)
        depth_all = (depth_all - depth_all.min(dim=1, keepdim=True)[0]) / (depth_all.max(dim=1, keepdim=True)[0] - depth_all.min(dim=1, keepdim=True)[0])
        #activity_mask = torch.ones_like(activity_mask)
        activity_mask = (depth_all < 0.5).float().unsqueeze(-1)
        if inputs['epoch'] == 0 or (inputs['epoch']+1) % self.opt['active_occupancy_refinement_epochs'] == 0:
            self.activity_occupancy = occupancy_field.detach().mean(dim=0)
            self.activity_occupancy_rgb = rgb_field.detach().mean(dim=0)
        if self.ao is None:
            self.ao_count = activity_mask.sum(dim=0)
            self.ao = (occupancy_field.detach() * activity_mask).sum(dim=0) 
            #self.activity_occupancy = torch.ones_like(self.ao)   # First epoch suppose they are all active on average. (if the canonical
                                                                 # consistency loss it is turned on, they should be all active on average
                                                                 # from previous refinement epochs)
            self.ao_rgb = (rgb_field.detach() * activity_mask).sum(dim=0) 
            #self.activity_occupancy_rgb = torch.ones_like(self.ao_rgb)
        else:
            # Contribute to activity occupancy and ao_rgb only with values towards the front, se we correct the further values
            '''
            depth_all = torch.norm(dynamical_voxels_world_coo - self.cam_loc.unsqueeze(1), dim=-1)
            depth_all = (depth_all - depth_all.min(dim=1, keepdim=True)[0]) / (depth_all.max(dim=1, keepdim=True)[0] - depth_all.min(dim=1, keepdim=True)[0])
            weights = ((1 - depth_all)).unsqueeze(-1)
            activity_occupancy = occupancy_field.clone().detach() + (self.activity_occupancy.expand_as(occupancy_field) - occupancy_field.clone().detach()) * (1 - weights)
            ao_rgb = rgb_field.clone().detach() + (self.ao_rgb.expand_as(rgb_field) - rgb_field.clone().detach()) * (1 - weights)
            self.activity_occupancy = (self.activity_occupancy + activity_occupancy.mean(dim=0).unsqueeze(0)) / 2
            self.ao_rgb = (self.ao_rgb + ao_rgb.mean(dim=0).unsqueeze(0)) / 2
            '''
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

            #self.activity_occupancy = (self.activity_occupancy * 9 + occupancy_field.detach().mean(dim=0).unsqueeze(0)) / 10
            #self.ao_rgb = (self.ao_rgb * 9 + rgb_field.detach().mean(dim=0).unsqueeze(0)) / 10
            #self.activity_occupancy = torch.clamp(self.activity_occupancy + occupancy_field.detach().mean(dim=0).unsqueeze(0) - 0.1, 0, 1)
            #self.ao_rgb = torch.clamp(self.ao_rgb + rgb_field.detach().mean(dim=0).unsqueeze(0) - 0.1, 0, 1)

        if batch_idx == num_samples - 1:    
            if self.visualize_stats:
                with torch.no_grad():
                    thr_mean = torch.quantile(self.activity_occupancy, 0.5)
                    #activity_occupancy_map = (self.activity_occupancy > min(self.opt['ao_threshold'], thr_mean)).squeeze(-1)
                    activity_occupancy_map = (self.activity_occupancy > self.opt['ao_threshold']).squeeze(-1)
                    try:
                        activity_occupancy_voxels_coos = active_voxels[activity_occupancy_map]
                        activity_voxels_ovs = self.activity_occupancy[activity_occupancy_map]
                        activity_voxels_rgbs = self.activity_occupancy_rgb[activity_occupancy_map]
                        self.visualize_voxels(activity_occupancy_voxels_coos, output_file='activity_occupancy_voxels_coo.png', world=True, voxels_ov=activity_voxels_ovs, voxels_rgb=activity_voxels_rgbs)
                    except:
                        pass
        
        if self.prev_of is None:
            self.prev_of = occupancy_field.detach()
            self.prev_rgb = rgb_field.detach()

        outputs = {'image': image,
                   'dynamical_voxels_coo': dynamical_voxels_world_coo,
                   'occupancy_field': occupancy_field,
                   'occupancy_field_t': occupancy_field_t,
                   'softmin': softmin,
                   'ray_opacity': ray_opacity,
                   'outliers_mask': outliers_mask,
                   #'occupancy_field_grad': occupancy_field_grad,
                   'cum_of': cum_of,
                   'of_dpt': of_dpt,
                   'of_diff': of_diff,
                   'depth_image': depth_image,
                   'rgb_field': rgb_field, 
                   #'rgb_field_t': rgb_field_t,   # This might not be needed when shadow field is present
                   'voxels_rgb_t': voxels_rgb_t,
                   'rasterized_image': rasterized_image,
                   #'shadow_field': shadow_field,
                   'activity_occupancy': self.activity_occupancy,
                   'activity_occupancy_rgb': self.activity_occupancy_rgb,
                   'prev_of': self.prev_of,
                   'prev_rgb': self.prev_rgb,
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

        self.prev_of = occupancy_field.detach()
        self.prev_rgb = rgb_field.detach()

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
        thr_mean = torch.quantile(self.activity_occupancy, 0.1)
        ao_map = (activity_occupancy > min(self.opt['ao_threshold'], thr_mean)).float().squeeze(-1)
        #ao_map = (activity_occupancy > self.opt['ao_threshold']).float().squeeze(-1)
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
            occ_mm_np = binary_dilation(occ_mm_np, structure=np.ones((3,3,3)), iterations=2)

            occ_mm_np = binary_erosion(occ_mm_np, structure=np.ones((3,3,3)), iterations=2)
            occupied_matrix_mapping = torch.from_numpy(occ_mm_np).float().cuda()

            # Increase resolution
            sf = (n_active_voxels_prev / n_active_voxels_post)**(1/3)
            print("scale factor: ", sf)
            occupied_matrix_mapping = upsample_matrix(occupied_matrix_mapping, scale_factor=sf.item(), threshold=0.4)

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
    def initialize_matrix_mapping(self, mmap_dim, mmap_res):
        '''This method is used to initialize the matrix_mapping with random positions close to the center.
        '''
        matrix_mapping = torch.ones(mmap_dim, mmap_dim, mmap_dim).cuda()
        active_voxels = self.voxel_mapping(matrix_mapping)
        
        # Load info for one frame to obtain canonical inliers and outliers
        if True:
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
            metadata_path = os.path.join(canonical_init_video_folder, "metadata.yaml")
            with open(metadata_path, 'r') as file:
                metadata = yaml.safe_load(file)
            gender = metadata['gender']

            # Obtain canonical inliers and compute corresponding matrix mapping 
            deformer = SMPLDeformer(betas=betas, gender=gender)
            x_d, outlier_mask = deformer.forward(active_voxels, smpl_tfs, return_weights=False, inverse=False, smpl_verts=smpl_verts_cano.unsqueeze(0))
            ao_map = (~outlier_mask).float()
            occupied_matrix_mapping = self.voxel_mapping(matrix_mapping, active_voxels, ao_map)   # The matrix mapping is initialized with the occupied voxels
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

    def update_matrix_mapping(self, matrix_mapping, occupancy_field):
        '''This method is used to update the matrix_mapping based on the occupancy field.
        '''
        return matrix_mapping

    def interpolate_to_voxels(self, features, voxels_coo, voxels_uv, rgb=False):
        '''This method is used to interpolate the features to the voxels.'''
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
        interpolated_features = torch.zeros(self.batch_size, voxels_coo.shape[1], features.shape[1]).to(features.device)
        #interpolated_features[..., :] = features[torch.arange(features.shape[0]).unsqueeze(1), :, y_int, x_int]
        x_frac = x_frac.unsqueeze(-1).repeat(1, 1, features.shape[1])
        y_frac = y_frac.unsqueeze(-1).repeat(1, 1, features.shape[1])
        interpolated_features = features[batch_indices, :, y_int, x_int] * x_frac * y_frac + \
                                        features[batch_indices, :, y_int, torch.clamp(x_int+1, 0, features.shape[3] - 1)] * (1 - x_frac) * y_frac + \
                                        features[batch_indices, :, torch.clamp(y_int+1, 0, features.shape[2] - 1), x_int] * x_frac * (1 - y_frac) + \
                                        features[batch_indices, :, torch.clamp(y_int+1, 0, features.shape[2] - 1), torch.clamp(x_int+1, 0, features.shape[3] - 1)] * (1 - x_frac) * (1 - y_frac)

        if not rgb:
            depth_values = torch.norm(voxels_coo - self.cam_loc.unsqueeze(1), dim=-1)
            depth_values = (depth_values - depth_values.min(dim=1, keepdim=True)[0]) / (depth_values.max(dim=1, keepdim=True)[0] - depth_values.min(dim=1, keepdim=True)[0]) 
            z_int = (depth_values * (features.shape[1] - 1)).long()
            z_frac = (depth_values * (features.shape[1] - 1) - z_int.float())
            #features = interpolated_features[torch.arange(interpolated_features.shape[0]).unsqueeze(1), torch.arange(interpolated_features.shape[1]), z_int]
            voxels_indices = torch.arange(interpolated_features.shape[1])
            features = interpolated_features[batch_indices, voxels_indices, z_int] * z_frac + \
                        interpolated_features[batch_indices, voxels_indices, torch.clamp(z_int+1, max=features.shape[1]-1)] * (1 - z_frac)
            features = features.unsqueeze(-1)
        if rgb:
            depth_values = torch.norm(voxels_coo - self.cam_loc.unsqueeze(1), dim=-1)
            depth_values = (depth_values - depth_values.min(dim=1, keepdim=True)[0]) / (depth_values.max(dim=1, keepdim=True)[0] - depth_values.min(dim=1, keepdim=True)[0]) 
            z_r_int = (depth_values * (int(features.shape[1]//3) - 1)).long()
            z_g_int = (depth_values * (int(features.shape[1]//3) - 1)).long() + features.shape[1]//3
            z_b_int = (depth_values * (int(features.shape[1]//3) - 1)).long() + 2*features.shape[1]//3
            z_r_frac = (depth_values * (int(features.shape[1]//3) - 1) - z_r_int.float())
            z_g_frac = (depth_values * (int(features.shape[1]//3) - 1) + features.shape[1]//3 - z_g_int.float())
            z_b_frac = (depth_values * (int(features.shape[1]//3) - 1) + 2*features.shape[1]//3 - z_b_int.float())
            #r = interpolated_features[torch.arange(interpolated_features.shape[0]).unsqueeze(1), torch.arange(interpolated_features.shape[1]), z_r_int].unsqueeze(-1)
            #g = interpolated_features[torch.arange(interpolated_features.shape[0]).unsqueeze(1), torch.arange(interpolated_features.shape[1]), z_g_int].unsqueeze(-1)
            #b = interpolated_features[torch.arange(interpolated_features.shape[0]).unsqueeze(1), torch.arange(interpolated_features.shape[1]), z_b_int].unsqueeze(-1)
            voxels_indices = torch.arange(interpolated_features.shape[1])
            r = interpolated_features[batch_indices, voxels_indices, z_r_int] * z_r_frac + \
                interpolated_features[batch_indices, voxels_indices, torch.clamp(z_r_int+1, max=(int(features.shape[1]//3) - 1))] * (1 - z_r_frac)
            g = interpolated_features[batch_indices, voxels_indices, z_g_int] * z_g_frac + \
                interpolated_features[batch_indices, voxels_indices, torch.clamp(z_g_int+1, max=(int(features.shape[1]//3) - 1)+features.shape[1]//3)] * (1 - z_g_frac)
            b = interpolated_features[batch_indices, voxels_indices, z_b_int] * z_b_frac + \
                interpolated_features[batch_indices, voxels_indices, torch.clamp(z_b_int+1, max=(int(features.shape[1]//3) - 1)+2*features.shape[1]//3)] * (1 - z_b_frac)
            r = r.unsqueeze(-1)
            g = g.unsqueeze(-1)
            b = b.unsqueeze(-1)
            features = torch.cat((r, g, b), dim=-1)

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
                smpl_verts = self.smpl_verts_cano
                deformer = SMPLDeformer(betas=betas[i], gender=gender[i])
                x_d[i], outlier_mask = deformer.forward(points_coo[i], smpl_tfs[i], return_weights=False, inverse=False, smpl_verts=smpl_verts)
                #x_d[i], outlier_mask = deformer.forward(points_coo[i] * 1.1, smpl_tfs[i], return_weights=False, inverse=False, smpl_verts=smpl_verts)
            if self.visualize_stats:
                with torch.no_grad():
                    self.visualize_voxels(points_coo[0][~outlier_mask], output_file='canonical_non_outliers.png', world=False)
                    self.visualize_voxels(x_d[0][~outlier_mask], output_file='dynamical_non_outliers.png', world=True)
        return x_d

    def canonical_tfs_from_dynamical(self, points_coo, smpl_tfs, betas, gender, smpl_outputs, batch=False):
        if batch:
            x_c = torch.zeros(self.batch_size, points_coo.shape[1], points_coo.shape[2]).cuda().float()
            x_d = torch.zeros(self.batch_size, points_coo.shape[1], points_coo.shape[2]).cuda().float()
            for i in range(self.batch_size):
                deformer = SMPLDeformer(betas=betas[i], gender=gender[i])
                x_c[i], outlier_mask = deformer.forward(points_coo[i].clone(), smpl_tfs[i], return_weights=False, inverse=True, smpl_verts=smpl_outputs['smpl_verts'][i])
                x_d[i], outlier_mask = deformer.forward(x_c[i], smpl_tfs[i], return_weights=False, inverse=False, smpl_verts=smpl_outputs['smpl_verts'][i])
            if self.visualize_stats:
                with torch.no_grad():
                    self.visualize_voxels(points_coo[0], output_file='dynamical_after_transform.png', world=True)
                    self.visualize_voxels(points_coo[0][~outlier_mask], output_file='dynamical_non_outliers.png', world=True)
                    self.visualize_voxels(x_c[0][~outlier_mask], output_file='canonical_non_outliers.png', world=True)
                    self.visualize_voxels(x_c[0], output_file='canonical_after_transform.png', world=True)
                    self.visualize_voxels(x_d[0], output_file='dynamical_after_inverse.png', world=True)
        return x_c
    
    def generate_mesh(self, voxels_coo, occupancy_field, voxels_rgb, outputs_folder, frame_id=None, mode='cano'):
        '''This method is used to generate a mesh from the voxels.
        '''
        with torch.no_grad():
            if mode == 'cano':   # Canonical shape from single frame predictions
                for i in range(self.batch_size):
                    t0_mesh = time.time()
                    ov_mask = (occupancy_field[i] > self.occupancy_threshold).squeeze(-1).float()
                    active_voxels = self.voxel_mapping(self.matrix_mapping)
                    voxels_matrix = self.voxel_mapping(self.matrix_mapping, voxels_coo=active_voxels, voxels_values=ov_mask)
                    
                    verts, faces, normals, values = measure.marching_cubes(voxels_matrix.cpu().numpy())
                    
                    rgb_values = voxels_rgb[i][ov_mask > 0.5]
                    rgb_values = torch.cat((rgb_values, occupancy_field[i][ov_mask > 0.5]), dim=-1).cpu().numpy()
                    rgb_values = (rgb_values * 255).astype(np.uint8)
                    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=rgb_values)
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
                assert voxels_coo is None, "voxels_coo should be None for active occupancy canonical reconstruction"
                t0_mesh = time.time()
                ov_mask = (occupancy_field > self.opt['ao_threshold']).squeeze(-1).float()
                active_voxels = self.voxel_mapping(self.matrix_mapping)   # Retrieve cano coords from matrix mapping
                voxels_matrix = self.voxel_mapping(self.matrix_mapping, active_voxels, ov_mask)
                #voxels_matrix = upsample_matrix(voxels_matrix, scale_factor=2, threshold=0.5)
                
                verts, faces, normals, values = measure.marching_cubes(voxels_matrix.cpu().numpy(), level=0.5)
                
                rgb_values = voxels_rgb[ov_mask.bool()]
                rgb_values = torch.cat((rgb_values, occupancy_field[ov_mask.bool()]), dim=-1).cpu().numpy()
                rgb_values = (rgb_values * 255).astype(np.uint8)

                if verts.shape[0] != rgb_values.shape[0]:
                    raise ValueError(f"Number of vertices ({verts.shape[0]}) and RGB values ({rgb_values.shape[0]}) do not match")

                mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=rgb_values)
                #mesh.fill_holes()

                if frame_id is None:
                    output_path = 'outputs/mesh.ply'
                else:
                    output_path = os.path.join(outputs_folder, f'mesh_epoch_{frame_id}.ply')
                mesh.export(output_path)
                t1_mesh = time.time()
                wandb.log({"Time for mesh generation": t1_mesh-t0_mesh})
            elif mode == 'dynamical_pc':   # Point cloud reconstruction in the dynamical space
                assert voxels_coo is not None, "voxels_coo is needed for the dynamical space reconstruction"
                voxels_coo = voxels_coo[0]   # Remove batch dimension
                assert occupancy_field.shape[1] == 1, "occupancy_field and rest of the sample should not be batched"
                t0_mesh = time.time()
                ov_mask = (occupancy_field > 0.5).squeeze(-1).float()

                voxels_coo = voxels_coo[ov_mask.bool()].cpu().numpy()
                                
                rgb_values = voxels_rgb[ov_mask.bool()]
                rgb_values = torch.cat((rgb_values, occupancy_field[ov_mask.bool()]), dim=-1).cpu().numpy()

                point_cloud = pv.PolyData(voxels_coo)
                point_cloud['colors_with_opacity'] = rgb_values

                mesh = point_cloud.delaunay_3d()
                surface_mesh = mesh.extract_surface()
                surface_mesh['colors_with_opacity'] = point_cloud['colors_with_opacity']
                #surface_mesh['RGBA'] = point_cloud['RGBA']

                if frame_id is None:
                    output_path = 'outputs/mesh.ply'
                else:
                    output_path = os.path.join(outputs_folder, f'dynamical_epoch_{frame_id}.ply')
                surface_mesh.save(output_path)
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

    def visualize_voxels(self, voxels_coo, output_file=None, world=False, voxels_ov=None, voxels_rgb=None):
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
            plt.close()

        return