import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DoubleConv, DownSample, UpSample
from deformer import SMPLDeformer, skinning
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from torchvision.utils import make_grid, save_image
from torchvision import transforms
import time
from utils import get_camera_loc, coordinates_projection_map, get_boundaries, get_rays, get_uv
from collections import OrderedDict
import wandb

import cv2
from torchvision.transforms import Compose
import sys
sys.path.append('/UnA-Gen/supp_repos/Depth_Anything_main/')
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 16)
        self.down_convolution_2 = DownSample(16, 32)
        self.down_convolution_3 = DownSample(32, 64)
        #self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(64, 128)

        #self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(128, 64)
        self.up_convolution_3 = UpSample(64, 32)
        self.up_convolution_4 = UpSample(32, 16)
        
        self.out = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=1)

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

class OccupancyField_fc(nn.Module):
    def __init__(self, in_features=35, num_classes=1):
        super(OccupancyField_fc, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features-3),
            nn.ReLU(inplace=True),
            nn.Linear(in_features-3, in_features-3),
            nn.ReLU(inplace=True),
            nn.Linear(in_features-3, in_features-3),
            nn.ReLU(inplace=True),
            nn.Linear(in_features-3, (in_features-3)//2),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//2, (in_features-3)//2),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//2, (in_features-3)//2),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//2, (in_features-3)//4),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//4, (in_features-3)//4),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//4, (in_features-3)//4),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//4, (in_features-3)//8),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//8, (in_features-3)//8),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//8, (in_features-3)//8),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//8, (in_features-3)//16),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//16, (in_features-3)//16),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//16, (in_features-3)//16),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//16, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
class RGBfield_fc(nn.Module):
    def __init__(self, in_features=35, num_classes=3):
        super(RGBfield_fc, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features-3),
            nn.ReLU(inplace=True),
            nn.Linear(in_features-3, in_features-3),
            nn.ReLU(inplace=True),
            nn.Linear(in_features-3, in_features-3),
            nn.ReLU(inplace=True),
            nn.Linear(in_features-3, (in_features-3)//2),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//2, (in_features-3)//2),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//2, (in_features-3)//2),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//2, (in_features-3)//4),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//4, (in_features-3)//4),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//4, (in_features-3)//4),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//4, (in_features-3)//8),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//8, (in_features-3)//8),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//8, (in_features-3)//8),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//8, (in_features-3)//16),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//16, (in_features-3)//16),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//16, (in_features-3)//16),
            nn.ReLU(inplace=True),
            nn.Linear((in_features-3)//16, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class OccupancyField_cnn(nn.Module):
    def __init__(self, in_features=35, depth_levels=1):
        super(OccupancyField_cnn, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=depth_levels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return torch.sigmoid(8 * x)

class RGBfield_cnn(nn.Module):
    def __init__(self, in_features=35, depth_levels=3, view_dependent=False):
        super(RGBfield_cnn, self).__init__()

        self.view_dependent = view_dependent
        self.depth_levels = depth_levels

        if not view_dependent:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            )
        elif view_dependent:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=9, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=9, out_channels=9, kernel_size=1, stride=1, padding=0),                
                nn.ReLU(inplace=True),
            )
            self.conv_r = nn.Sequential(
                nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=9, out_channels=9, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=9, out_channels=6, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=1, stride=1, padding=0), 
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=3, out_channels=depth_levels, kernel_size=3, stride=1, padding=1),         
            )
            self.conv_g = nn.Sequential(
                nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=9, out_channels=9, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=9, out_channels=6, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=1, stride=1, padding=0), 
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=3, out_channels=depth_levels, kernel_size=3, stride=1, padding=1),         
            )
            self.conv_b = nn.Sequential(
                nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=9, out_channels=9, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=9, out_channels=6, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=1, stride=1, padding=0), 
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=3, out_channels=depth_levels, kernel_size=3, stride=1, padding=1),         
            )

    def forward(self, x):
        if self.view_dependent:
            x = self.conv(x)
            x_r = self.conv_r(x)
            x_g = self.conv_g(x)
            x_b = self.conv_b(x)
            x = torch.cat((x_r, x_g, x_b), dim=1)
        else:
            x = self.conv(x)
        return torch.sigmoid(2 * x)
    
class cOccupancyField_cnn(nn.Module):
    def __init__(self, in_features=35, out_channels=1):
        super(cOccupancyField_cnn, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),     
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class cRGBfield_cnn(nn.Module):
    def __init__(self, in_features=35, out_channels=3, view_dependent=False):
        super(cRGBfield_cnn, self).__init__()

        self.view_dependent = view_dependent
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.conv_r = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=3, stride=1, padding=1), 
            nn.Sigmoid(),           
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=3, stride=1, padding=1),       
            nn.Sigmoid(),       
        )
        self.conv_b = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=3, stride=1, padding=1), 
            nn.Sigmoid(),        
        )

    def forward(self, x):
        x = self.conv(x)
        x_r = self.conv_r(x)
        x_g = self.conv_g(x)
        x_b = self.conv_b(x)
        x = torch.cat((x_r, x_g, x_b), dim=1)

        return x
    
class worldParam(nn.Module):
    def __init__(self, in_features=35, num_classes=6):
        super(worldParam, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(478632, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 6),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
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
        self.intrinsics = None   
        self.smpl_params = None
        self.pose = None
        self.cam_loc = None
        self.fov_bounds = None
        self.scale = None
        self.trans = None
        self.standard_depth_n = torch.nn.Parameter(torch.tensor(opt['standard_depth_n'], dtype=torch.float32))
        self.standard_depth_f = opt['standard_depth_f']
        self.n_training_rays = opt['n_training_rays']
        self.mapping_dim = opt['mapping_dim']
        self.occupancy_threshold = opt['occupancy_threshold']
        self.matrix_mapping = self.initialize_matrix_mapping(self.mapping_dim, 0.5).cuda()
        self.mapping_prob_density = torch.zeros_like(self.matrix_mapping).cuda()

        self.visualize_stats = opt['visualize_stats']

        # Unet encoder initialization
        if opt['encoder']['name'] == 'unet':
            if opt['encoder']['decouple']: 
                self.GeometryEncoder = UNetEncoder(in_channels=in_channels, out_channels=features).cuda()
                self.RenderingEncoder = UNetEncoder(in_channels=in_channels, out_channels=features).cuda()
                if not opt['encoder']['train']:
                    self.GeometryEncoder.eval()
                    self.RenderingEncoder.eval()
                    for param in self.GeometryEncoder.parameters():
                        param.requires_grad = False
                    for param in self.RenderingEncoder.parameters():
                        param.requires_grad = False
            else:
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

        '''
        for param in self.GeometryEncoder.parameters():
            param.requires_grad = False

        for param in self.RenderingEncoder.parameters():
            param.requires_grad = False
        '''
        
        if self.opt['learn_world_params']:
            self.worldParams = worldParam(in_features=features+3, num_classes=6).cuda()

        if self.opt['decoder']['name'] == "fc":
            self.OccupancyField = OccupancyField_fc(in_features=features+3, num_classes=1).cuda()
            self.RGBField = RGBfield_fc(in_features=features+3, num_classes=3).cuda()
        elif self.opt['decoder']['name'] == "cnn":
            if self.opt['decoder']['canonical']:
                self.OccupancyField = cOccupancyField_cnn(in_features=features+3, out_channels=self.mapping_dim).cuda()
            else:
                if self.opt['decoder']['occupancy_depth_dep']:
                    self.OccupancyField = OccupancyField_cnn(in_features=features+3, depth_levels=10).cuda()
                else:
                    self.OccupancyField = OccupancyField_cnn(in_features=features+3, depth_levels=1).cuda()
            if self.opt['decoder']['view_dependent_rgb']:
                if self.opt['decoder']['canonical']:
                    self.RGBField = cRGBfield_cnn(in_features=features+3, out_channels=self.mapping_dim).cuda()
                else:
                    if self.opt['decoder']['rgb_depth_dep']:
                        self.RGBField = RGBfield_cnn(in_features=features+3, depth_levels=3, view_dependent=True).cuda()    
                    else:
                        self.RGBField = RGBfield_cnn(in_features=features+3, depth_levels=1, view_dependent=True).cuda()
            else:
                self.RGBField = RGBfield_cnn(in_features=features+3, depth_levels=1, view_dependent=False).cuda()

        self.apply(self.weights_init)   # weights initialization

    def forward(self, inputs):
        # input is a dictionary containing the image, the smpl parameters, and other info 
        # it is of shape (batch_size, key_values) where key values is of different dimensions
        t0 = time.time()  
        image = inputs['masked_image']
        self.img_size = (image.shape[2], image.shape[3])
        self.batch_size = image.shape[0]
        # resize_transform = transforms.Resize((512, 512))
        # image = resize_transform(original_image)
        depth_image = inputs['depth_image']

        gender = inputs['metadata']['gender']
        betas = inputs['betas']
        # scale = inputs['smpl_params'][:, 0].cuda()
        self.pose = inputs["pose"].cuda()
        self.intrinsics = inputs['intrinsics'].cuda()
        self.smpl_params = inputs['smpl_params'].cuda()
        smpl_tfs = inputs['smpl_tfs'].cuda()
        self.scale = torch.tensor(1.0).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, 3).cuda()
        self.trans = torch.tensor(0.0).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, 3).cuda()

        matrix_mapping = self.matrix_mapping

        with torch.no_grad():
            # Learn normalization transformation for the metric spaces
            # I <> D <> C   I: Image space, D: Dynamical space (metric), C: Canonical space (metric)
            #      |    |
            #      D    C   D: Dynamical space (normalized), C: Canonical space (normalized)
            # The metric spaces are used to establish the 1-1 correspondence by skinning
            # The normalized spaces are used to establish the 1-1 correspondence by the feature encoder and evaluate the networks
            self.fov_bounds, self.cam_loc = get_boundaries(image, self.intrinsics, self.pose)   # These parameters are used throught many methods

            # The active voxels are mapped by the matrix_mapping (while in the canonical space)
            # TODO: Implement the mapping of the active voxels by the matrix_mapping 
            canonical_voxels_coo = self.voxel_mapping(matrix_mapping)   # Returns the active voxels coordinates in the canonical space
            if self.visualize_stats:
                self.visualize_voxels(canonical_voxels_coo, output_file='active_voxels.png')

            # The active voxels are transformed to the dynamical space from the canonical space using the smpl pose parameters
            # TODO: Implement the transformation of the active voxels to the dynamical space using the smpl pose parameters
            # dynamical_voxels_coo = transform_voxels(active_voxels_coo, smpl_tfs)   # Returns the active voxels coordinates in the dynamical space

            if self.opt['smpl_transform'] == 'world':
                world_canonical_voxels_coo = canonical_voxels_coo.unsqueeze(0).repeat(self.batch_size, 1, 1)
                world_canonical_voxels_coo = self.norm_to_world_coo(world_canonical_voxels_coo, self.cam_loc, self.fov_bounds)
                dynamical_voxels_coo = self.dynamical_tfs_from_canonical(world_canonical_voxels_coo, smpl_tfs, betas, gender, batch=True)   # To be tested if it works
                dynamical_voxels_world_coo = dynamical_voxels_coo
                self.visualize_voxels(world_canonical_voxels_coo[0], output_file='canonical_voxels_world_coo.png', world=True)
            elif self.opt['smpl_transform'] == 'norm':
                dynamical_voxels_coo = self.dynamical_tfs_from_canonical(canonical_voxels_coo, smpl_tfs, betas, gender, batch=False)
                dynamical_voxels_world_coo = self.norm_to_world_coo(dynamical_voxels_coo, self.cam_loc, self.fov_bounds)
            
            voxels_uv, _, _ = get_uv(dynamical_voxels_world_coo, self.cam_loc, self.intrinsics, self.pose)
                
            if self.visualize_stats:
               self.visualize_voxels(dynamical_voxels_coo[0], output_file='dynamical_voxels.png', world=True)
            # Normalize the coordinates to the [-1, 1] range. The transformation shoots points out. We should figure out why, and fix it.
            #dynamical_voxels_coo[:, :, 0] = (dynamical_voxels_coo[:, :, 0] - dynamical_voxels_coo[:, :, 0].min(dim=1, keepdim=True)[0]) / (dynamical_voxels_coo[:, :, 0].max(dim=1, keepdim=True)[0] - dynamical_voxels_coo[:, :, 0].min(dim=1, keepdim=True)[0]) * 2 - 1  
            #dynamical_voxels_coo[:, :, 1] = (dynamical_voxels_coo[:, :, 1] - dynamical_voxels_coo[:, :, 1].min(dim=1, keepdim=True)[0]) / (dynamical_voxels_coo[:, :, 1].max(dim=1, keepdim=True)[0] - dynamical_voxels_coo[:, :, 1].min(dim=1, keepdim=True)[0]) * 2 - 1
            #dynamical_voxels_coo[:, :, 2] = (dynamical_voxels_coo[:, :, 2] - dynamical_voxels_coo[:, :, 2].min(dim=1, keepdim=True)[0]) / (dynamical_voxels_coo[:, :, 2].max(dim=1, keepdim=True)[0] - dynamical_voxels_coo[:, :, 2].min(dim=1, keepdim=True)[0]) * 2 - 1

        # ---
        # Trained unet encoder
        # The feature encoder produces a pixel/voxel-wise feature embedding used to compute the occupancy field of the active voxels
        if self.opt['encoder']['train']:   # The encoder is trained 
            if self.opt['encoder']['name'] == 'unet':
                if self.opt['encoder']['decouple']:
                    features = self.GeometryEncoder(image)   # (batch_size, features, img_width, img_height)
                    features_r = self.RenderingEncoder(image)
                else:
                    features = self.encoder(image)
                    features_r = features

            if self.opt['encoder']['name'] == 'dpt':
                transform = Compose([
                    Resize(
                        width=518,
                        height=518,
                        resize_target=False,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=14,
                        resize_method='lower_bound',
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    #NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   # Already normalized in dataloader
                    PrepareForNet(),
                ])

                #features = torch.zeros((self.batch_size, 32, 924, 518)).cuda()
                features = torch.zeros((self.batch_size, 128, 924, 518)).cuda()   # Change the number of features depending on the encoder size
                for i,im in enumerate(inputs['masked_img_path']):
                    raw_image_dpt = cv2.imread(im)
                    image_dpt = cv2.cvtColor(raw_image_dpt, cv2.COLOR_BGR2RGB) / 255.0
                    image_dpt = transform({'image': image_dpt})['image']
                    image_dpt = torch.from_numpy(image_dpt).unsqueeze(0).cuda()
                    test = self.dptEncoder(image_dpt)
                    #mean = test.mean()
                    #std = test.std()
                    #test_normalized = (test - mean) / std
                    features[i] = test   

                features_r = features

        elif not self.opt['encoder']['train']:   # Training of the encoder is paused
            with torch.no_grad():   
                if self.opt['encoder']['name'] == 'unet':
                    if self.opt['encoder']['decouple']:
                        features = self.GeometryEncoder(image)   # (batch_size, features, img_width, img_height)
                        features_r = self.RenderingEncoder(image)
                    else:
                        features = self.encoder(image)
                        features_r = features

                if self.opt['encoder']['name'] == 'dpt':
                    transform = Compose([
                        Resize(
                            width=518,
                            height=518,
                            resize_target=False,
                            keep_aspect_ratio=True,
                            ensure_multiple_of=14,
                            resize_method='lower_bound',
                            image_interpolation_method=cv2.INTER_CUBIC,
                        ),
                        #NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        PrepareForNet(),
                    ])
                    
                    #features = torch.zeros((self.batch_size, 32, 924, 518)).cuda()
                    features = torch.zeros((self.batch_size, 128, 924, 518)).cuda()   # Change the number of features depending on the encoder size
                    for i,im in enumerate(inputs['masked_img_path']):
                        raw_image_dpt = cv2.imread(im)
                        image_dpt = cv2.cvtColor(raw_image_dpt, cv2.COLOR_BGR2RGB) / 255.0
                        image_dpt = transform({'image': image_dpt})['image']
                        image_dpt = torch.from_numpy(image_dpt).unsqueeze(0).cuda()
                        test = self.dptEncoder(image_dpt)
                        #mean = test.mean()
                        #std = test.std()
                        #test_normalized = (test - mean) / std
                        features[i] = test   

                    features_r = features               

        # Infer world change of coordinates (batch_size, n_samples, 131)
        if self.opt['learn_world_params']:   # We need to compute this before if the smpl transform is done in world coo, as it's needed in world coo transform
            world_param = self.worldParams(features)
            self.scale = world_param[:, :3]
            self.trans = world_param[:, 3:]            
  
        if self.opt['decoder']['name'] == 'fc':
            features = self.interpolate_to_voxels(features, dynamical_voxels_world_coo, voxels_uv, canonical_voxels_coo)
            occupancy_field = self.OccupancyField(features.view(-1, features.shape[-1]))
            occupancy_field = occupancy_field.view(self.batch_size, features.shape[1], -1)
            
        elif self.opt['decoder']['name'] == 'cnn':
            occupancy_field = self.OccupancyField(features)
            if self.opt['decoder']['canonical']:
                occupancy_field = self.interpolate_to_voxels(occupancy_field, dynamical_voxels_world_coo, voxels_uv, canonical_voxels_coo, add_coo=False, int_to_cano='canonical')
            else:
                if self.opt['decoder']['occupancy_depth_dep']:    
                    occupancy_field = self.interpolate_to_voxels(occupancy_field, dynamical_voxels_world_coo, voxels_uv, canonical_voxels_coo, add_coo=False, depth_dep=True)
                else:
                    occupancy_field = self.interpolate_to_voxels(occupancy_field, dynamical_voxels_world_coo, voxels_uv, canonical_voxels_coo, add_coo=False, depth_dep=False)

        if self.opt['relative_ov']:
            eps = 1e-6
            occupancy_field = (occupancy_field - torch.min(occupancy_field, dim=1, keepdim=True)[0]) / (torch.max(occupancy_field, dim=1, keepdim=True)[0] - torch.min(occupancy_field, dim=1, keepdim=True)[0] + eps)

        if self.visualize_stats:
            with torch.no_grad():
                occupancy_field_0 = occupancy_field[0].view(-1).detach().cpu().numpy()
                wandb.log({"Occupancy field values distribution": wandb.Histogram(occupancy_field_0)})

        # --- Rendering ---
        if self.opt['decoder']['name'] == 'fc':
            features_r = self.interpolate_to_voxels(features_r, dynamical_voxels_world_coo, voxels_uv, canonical_voxels_coo, add_coo=True)

        rgb_field = self.RGBField(features_r)
        if self.opt['decoder']['name'] == 'fc':
            rgb_field = rgb_field.view(self.batch_size, features_r.shape[1], -1)
        elif self.opt['decoder']['name'] == 'cnn':
            if self.opt['decoder']['canonical']:
                rgb_field = self.interpolate_to_voxels(rgb_field, dynamical_voxels_world_coo, voxels_uv, canonical_voxels_coo, add_coo=False, rgb=True, int_to_cano='canonical')
            else:
                if self.opt['decoder']['rgb_depth_dep']:
                    rgb_field = self.interpolate_to_voxels(rgb_field, dynamical_voxels_world_coo, voxels_uv, canonical_voxels_coo, add_coo=False, depth_dep=True, rgb=True)
                else:
                    rgb_field = self.interpolate_to_voxels(rgb_field, dynamical_voxels_world_coo, voxels_uv, canonical_voxels_coo, add_coo=False, depth_dep=False, rgb=True)

        # ---
        if self.visualize_stats:
            with torch.no_grad():
                rgb_field_0 = rgb_field[0].detach().cpu().numpy()
                wandb.log({"RGB field values distribution": wandb.Histogram(rgb_field_0)})

        # The output of the occupancy field is used to compute the volume if needed
        # TODO: Implement the computation of the volume using the occupancy field as a separate method of the model, which can be called from outside

        # Also, the output of the occupancy field is used to refine the active voxels by modifying the matrix_mapping
        # The matrix mapping is to be kept larger than the actually occupied voxels, to allow for garment details etc.
        # However, where the active voxels are repeatedly not occupied, the matrix_mapping is to be reduced
        # The resolution (number of active voxels) reduces, while the number of occupied voxels remains the same (propotionally)
        # When the resolution is reduced enough, the matrix_mapping is to be refined to smaller voxels
        # TODO: implement the refinement of the matrix_mapping using the occupancy field as a separate method of the model, which can be called from outside
        # matrix_mapping = self.update_matrix_mapping(matrix_mapping, occupancy_field)
        # This update may be better done in the train script
        
        # render rgb values for optimization
        t0_render_rgb_values = time.time()
        original_rgb_values, rendered_rgb_values = self.render_values_at_rays(dynamical_voxels_world_coo, occupancy_field, rgb_field, image)
        t1_render_rgb_values = time.time()

        if self.visualize_stats:
            with torch.no_grad():
                wandb.log({"Time for rendering rgb values": (t1_render_rgb_values - t0_render_rgb_values)})

        outputs = {'dynamical_voxels_coo': dynamical_voxels_world_coo, 
                   'occupancy_field': occupancy_field, 
                   'rgb_field': rgb_field,
                   'original_rgb_values': original_rgb_values, 
                   'rendered_rgb_values': rendered_rgb_values}

        t1 = time.time()
        if self.visualize_stats:
            with torch.no_grad():
                wandb.log({"Time for forward pass": t1-t0})
            
        if False:
            ren_t0 = time.time()
            original_image, rendered_image = self.render_image(dynamical_voxels_coo, occupancy_field, rgb_field, image)
            ren_t1 = time.time()
            print(f'Rendering took {ren_t1-ren_t0} seconds')
            original_image = original_image.view(2, 3, 1920, 1080)
            rendered_image = rendered_image.view(2, 3, 1920, 1080)
            save_image(original_image, 'outputs/stats/original_image.png')
            save_image(rendered_image, 'outputs/stats/rendered_image.png')
        
        return outputs
    
    # Matrix mapping refinement -------------------------------------------------------------------------------------------
    def refinement(self, inputs):
        '''This method is used to refine the matrix_mapping based on the depth images.
        '''

        with torch.no_grad():
            image = inputs['masked_image']
            self.img_size = (image.shape[2], image.shape[3])
            self.batch_size = image.shape[0]
            # resize_transform = transforms.Resize((512, 512))
            # image = resize_transform(original_image)
            depth_image = inputs['depth_image'].cuda()

            gender = inputs['metadata']['gender']
            betas = inputs['betas']
            # scale = inputs['smpl_params'][:, 0].cuda()
            self.pose = inputs["pose"].cuda()
            self.intrinsics = inputs['intrinsics'].cuda()
            self.smpl_params = inputs['smpl_params'].cuda()
            smpl_tfs = inputs['smpl_tfs'].cuda()
            self.scale = torch.tensor(1.0).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, 3).cuda()
            self.trans = torch.tensor(0.0).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, 3).cuda()

            matrix_mapping = self.matrix_mapping

            self.fov_bounds, self.cam_loc = get_boundaries(image, self.intrinsics, self.pose)

            # Learn normalization transformation for the metric spaces
            # I <> D <> C   I: Image space, D: Dynamical space (metric), C: Canonical space (metric)
            #      |    |
            #      D    C   D: Dynamical space (normalized), C: Canonical space (normalized)
            # The metric spaces are used to establish the 1-1 correspondence by skinning
            # The normalized spaces are used to establish the 1-1 correspondence by the feature encoder and evaluate the networks

            # We select the rays corresponding to the spanning of our dynamical space with the active voxels
            '''
            u_span = torch.linspace(2, image_width-2, self.mapping_dim).cuda()   # Here we pad a bit, for the features to be interpolated
            v_span = torch.linspace(2, image_height-2, self.mapping_dim).cuda()   # and also to avoid waste. This parameter can be learned TODO
            u_span, v_span = torch.meshgrid(u_span, v_span)
            uv = torch.stack((u_span.flatten(), v_span.flatten()), dim=-1).float().cuda()
            ray_dirs, cam_loc = get_rays(uv, intrinsics, pose)
            '''

            # The active voxels are mapped by the matrix_mapping (while in the canonical space)
            # TODO: Implement the mapping of the active voxels by the matrix_mapping 
            canonical_voxels_coo = self.voxel_mapping(matrix_mapping)   # Returns the active voxels coordinates in the canonical space

            # The active voxels are transformed to the dynamical space from the canonical space using the smpl pose parameters
            # TODO: Implement the transformation of the active voxels to the dynamical space using the smpl pose parameters
            # dynamical_voxels_coo = transform_voxels(active_voxels_coo, smpl_tfs)   # Returns the active voxels coordinates in the dynamical space

            if self.opt['smpl_transform'] == 'world':
                world_canonical_voxels_coo = canonical_voxels_coo.unsqueeze(0).repeat(self.batch_size, 1, 1)
                world_canonical_voxels_coo = self.norm_to_world_coo(world_canonical_voxels_coo, self.cam_loc, self.fov_bounds)
                dynamical_voxels_coo = self.dynamical_tfs_from_canonical(world_canonical_voxels_coo, smpl_tfs, betas, gender, batch=True)   # To be tested if it works
            elif self.opt['smpl_transform'] == 'norm':
                dynamical_voxels_coo = self.dynamical_tfs_from_canonical(canonical_voxels_coo, smpl_tfs, betas, gender, batch=False)
                dynamical_voxels_coo = self.norm_to_world_coo(dynamical_voxels_coo, self.cam_loc, self.fov_bounds)

            dynamical_voxels_uv, _, _ = get_uv(dynamical_voxels_coo, self.cam_loc, self.intrinsics, self.pose)
            voxels_depth_values = torch.zeros(dynamical_voxels_coo.shape[0], dynamical_voxels_coo.shape[1], device=dynamical_voxels_coo.device)

            height, width = image.shape[-2:]
            for image_idx in range(image.shape[0]):
                for voxel_idx in range(dynamical_voxels_uv.shape[1]):
                    x, y = dynamical_voxels_uv[image_idx, voxel_idx]
                    y_int, x_int = y.long().item(), x.long().item()
                    y_frac, x_frac = y - y_int, x - x_int

                    voxels_depth_values[image_idx, voxel_idx] = depth_image[image_idx, :, y_int, x_int] * (y_frac * x_frac) + \
                                                            depth_image[image_idx, :, y_int, min(x_int+1, width-1)] * (y_frac * (1-x_frac)) + \
                                                            depth_image[image_idx, :, min(y_int+1, height-1), x_int] * ((1-y_frac) * x_frac) + \
                                                            depth_image[image_idx, :, min(y_int+1, height-1), min(x_int+1, width-1)] * ((1-y_frac) * (1-x_frac))

            voxels_depth_values = voxels_depth_values.view(dynamical_voxels_coo.shape[0], self.mapping_dim, self.mapping_dim, self.mapping_dim)
            for image_idx in range(image.shape[0]):
                self.mapping_prob_density += voxels_depth_values[image_idx]
            
        return
    
    # Auxiliary methods ---------------------------------------------------------------------------------------------------
    def initialize_matrix_mapping(self, mmap_dim, mmap_res):
        '''This method is used to initialize the matrix_mapping with random positions close to the center.
        '''
        matrix_mapping = torch.ones(mmap_dim, mmap_dim, mmap_dim)
        # TODO: implement the random selection of inner voxels

        return matrix_mapping
    
    def tan_angle_between(self, a, b):
        cross_product = torch.cross(a, b, dim=1)
        dot_product = torch.sum(a * b, dim=1)
        mag_cross_product = torch.norm(cross_product, dim=1)
        tan_theta = mag_cross_product / dot_product

        return tan_theta

    def norm_to_world_coo(self, voxels_coo, cam_loc, fov_bounds):
        '''Moves the voxels from the normed space to the world coordinates.
        The depth centered is a parameter that needs to be learned.
        '''
        central_ray = F.normalize(torch.mean(fov_bounds, dim=1), dim=1)
        x_m_ray = F.normalize(torch.mean(fov_bounds[:, -2:], dim=1), dim=1)
        y_m_ray = F.normalize(torch.mean(torch.stack((fov_bounds[:, 0], fov_bounds[:, 2]), dim=1), dim=1), dim=1)
        camera_x = F.normalize(x_m_ray - central_ray, dim=1)
        camera_y = F.normalize(y_m_ray - central_ray, dim=1)
        voxels_coo[:, :, 2] = voxels_coo[:, :, 2] * self.opt['standard_scale_z']
        x_scale_at_depth = (self.opt['standard_scale_z'] + self.standard_depth_n + voxels_coo[:,:,2]*torch.sign(central_ray.unsqueeze(1)[:,:,2]))*(self.tan_angle_between(central_ray, x_m_ray).unsqueeze(-1).expand_as(voxels_coo[:,:,2]))   # (batch_size, voxel) 
        y_scale_at_depth = (self.opt['standard_scale_z'] + self.standard_depth_n + voxels_coo[:,:,2]*torch.sign(central_ray.unsqueeze(1)[:,:,2]))*(self.tan_angle_between(central_ray, y_m_ray).unsqueeze(-1).expand_as(voxels_coo[:,:,2]))   # (batch_size, voxel)
        #body_center = self.smpl_params[:, 4:6].clone().unsqueeze(1)
        #print("smpl_params:", self.smpl_params[:, 4:6])
        #body_center[:,:,0] = (body_center[:,:,0]+1)/2*self.img_size[1]
        #body_center[:,:,1] = (body_center[:,:,1]+1)/2*self.img_size[0]
        #print("body_center:", body_center)
        #body_center_ray, _ = get_rays(body_center, self.intrinsics, self.pose)

        # Smpl estimated scaling
        #voxels_coo = voxels_coo * (self.smpl_params[:, 0].unsqueeze(1).unsqueeze(1).expand_as(voxels_coo))   # smpl estimated scale (batch_size)
        #voxels_coo[:, :, 0] = voxels_coo[:, :, 0] * x_scale_at_depth.unsqueeze(1).expand_as(voxels_coo[:, :, 0]) / self.smpl_params[:, 0].unsqueeze(1).expand_as(voxels_coo[:, :, 0])
        #voxels_coo[:, :, 1] = voxels_coo[:, :, 1] * y_scale_at_depth.unsqueeze(1).expand_as(voxels_coo[:, :, 1]) / self.smpl_params[:, 0].unsqueeze(1).expand_as(voxels_coo[:, :, 1])

        # Coordinates scaling
        if self.opt['distribute_with_perspective']:
            voxels_coo[:, :, 0] = voxels_coo[:, :, 0] * x_scale_at_depth
            voxels_coo[:, :, 1] = voxels_coo[:, :, 1] * y_scale_at_depth
        else:
            voxels_coo[:, :, 0] = voxels_coo[:, :, 0] * self.opt['standard_scale_z']
            voxels_coo[:, :, 1] = voxels_coo[:, :, 1] * self.opt['standard_scale_z']

        # Scaling: learnt
        #voxels_coo = voxels_coo * self.scale.unsqueeze(1).expand_as(voxels_coo)   # Scale (batch_size,3), affects only if learnt
        
        # Camera centered
        voxels_coo = voxels_coo + cam_loc.unsqueeze(1).expand_as(voxels_coo)  

        # Depth translation: base
        if self.opt['smpl_transform'] == 'world':
            voxels_coo = voxels_coo + central_ray.unsqueeze(1).expand_as(voxels_coo)*(self.opt['standard_scale_z'] + self.standard_depth_n) 
            #body_center = self.smpl_params[:, 4:6].clone().unsqueeze(1)
            #body_dir, _ = get_rays(body_center, self.intrinsics, self.pose)
            #voxels_coo = voxels_coo + body_dir.expand_as(voxels_coo)*(self.opt['standard_scale_z'] + self.standard_depth_n)
        elif self.opt['smpl_transform'] == 'norm':
            #voxels_coo = voxels_coo + central_ray.unsqueeze(1).expand_as(voxels_coo)*(self.opt['standard_scale_z'] + self.standard_depth_n) 
            body_center = self.smpl_params[:, 4:6].clone().unsqueeze(1)
            body_dir, _ = get_rays(body_center, self.intrinsics, self.pose)
            voxels_coo = voxels_coo + body_dir.expand_as(voxels_coo)*(self.opt['standard_scale_z'] + self.standard_depth_n)

        # Depth translation: learnt
        #voxels_coo = voxels_coo + central_ray.unsqueeze(1).expand_as(voxels_coo)*(1*self.scale.unsqueeze(1).expand_as(voxels_coo))   # Depth translation
        # smpl_params[:, 1:4] contain the relative position in the image plane
        
        # Smpl estimated translation
        #x_trans = self.smpl_params[:, 1]*x_scale_at_depth
        #y_trans = self.smpl_params[:, 2]*y_scale_at_depth
        #voxels_coo[:, :, 0] = voxels_coo[:, :, 0] + x_trans.unsqueeze(1).expand_as(voxels_coo[:, :, 0])
        #voxels_coo[:, :, 1] = voxels_coo[:, :, 1] + y_trans.unsqueeze(1).expand_as(voxels_coo[:, :, 1])
        #voxels_coo = voxels_coo + x_trans.unsqueeze(1).unsqueeze(-1).expand_as(voxels_coo)*camera_x.unsqueeze(1).expand_as(voxels_coo) + \
        #                        + y_trans.unsqueeze(1).unsqueeze(-1).expand_as(voxels_coo)*camera_y.unsqueeze(1).expand_as(voxels_coo)
        
        #voxels_coo = voxels_coo + body_center_ray.expand_as(voxels_coo)*(1 + self.standard_depth_n)  

        return voxels_coo
    
    def update_matrix_mapping(self, matrix_mapping, occupancy_field):
        '''This method is used to update the matrix_mapping based on the occupancy field.
        '''
        return matrix_mapping

    def visualize_voxels(self, voxels_coo, output_file=None, world=False, voxels_ov=None):
        '''This is used to save an image of the voxels in 3D space.
        '''
        with torch.no_grad():
            voxels_coo_np = voxels_coo.detach().cpu().numpy()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            if voxels_ov is not None: 
                voxels_ov_np = voxels_ov.detach().cpu().numpy()
                for i in range(voxels_coo_np.shape[0]):
                    ax.scatter(voxels_coo_np[i, 0], voxels_coo_np[i, 1], voxels_coo_np[i, 2], alpha=voxels_ov_np[i], color='blue')
            else:
                ax.scatter(voxels_coo_np[:, 0], voxels_coo_np[:, 1], voxels_coo_np[:, 2])

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

    def interpolate_to_voxels(self, features, voxels_coo, voxels_uv, canonical_voxels_coo=None, add_coo=True, depth_dep=False, rgb=False, int_to_cano=False):
        '''This method is used to interpolate the features to the voxels.'''
        if features.shape[2] == self.img_size[0] and features.shape[3] == self.img_size[1]:   # pixel-aligned features (unet encoder)
            x_int = voxels_uv[..., 0].long()
            y_int = voxels_uv[..., 1].long()
        else:
            x_int = ((voxels_uv[..., 0] / self.img_size[1]) * features.shape[3]).long()
            y_int = ((voxels_uv[..., 1] / self.img_size[0]) * features.shape[2]).long()
           
        # Use advanced indexing to gather features 
        if not int_to_cano:
            if add_coo:
                interpolated_features = torch.zeros(self.batch_size, voxels_coo.size(1), features.size(1) + voxels_coo.size(2)).to(features.device)
            else:
                interpolated_features = torch.zeros(self.batch_size, voxels_coo.size(1), features.size(1)).to(features.device)
            interpolated_features[..., :features.shape[1]] = features[torch.arange(features.shape[0]).unsqueeze(1), :, y_int, x_int]

            if add_coo and canonical_voxels_coo is not None:   # If the respective canonical voxels are provided, their canonical coordinates are passed
                interpolated_features[..., features.shape[1]:] = canonical_voxels_coo.unsqueeze(0).repeat(self.batch_size, 1, 1)
            elif add_coo:
                interpolated_features[..., features.shape[1]:] = voxels_coo
            else:
                pass

            if depth_dep:
                if not rgb:
                    depth_values = voxels_coo[:, :, 2]
                    depth_values = (depth_values - depth_values.min(dim=1, keepdim=True)[0]) / (depth_values.max(dim=1, keepdim=True)[0] - depth_values.min(dim=1, keepdim=True)[0]) 
                    z_int = (depth_values * (features.shape[1] - 1)).long()
                    features = interpolated_features[torch.arange(interpolated_features.shape[0]).unsqueeze(1), torch.arange(interpolated_features.shape[1]), z_int]
                    features = features.unsqueeze(-1)
                if rgb:
                    depth_values = voxels_coo[:, :, 2]
                    depth_values = (depth_values - depth_values.min(dim=1, keepdim=True)[0]) / (depth_values.max(dim=1, keepdim=True)[0] - depth_values.min(dim=1, keepdim=True)[0]) 
                    z_r_int = (depth_values * (int(features.shape[1]//3) - 1)).long()
                    z_g_int = (depth_values * (int(features.shape[1]//3) - 1)).long() + features.shape[1]//3
                    z_b_int = (depth_values * (int(features.shape[1]//3) - 1)).long() + 2*features.shape[1]//3
                    r = interpolated_features[torch.arange(interpolated_features.shape[0]).unsqueeze(1), torch.arange(interpolated_features.shape[1]), z_r_int].unsqueeze(-1)
                    g = interpolated_features[torch.arange(interpolated_features.shape[0]).unsqueeze(1), torch.arange(interpolated_features.shape[1]), z_g_int].unsqueeze(-1)
                    b = interpolated_features[torch.arange(interpolated_features.shape[0]).unsqueeze(1), torch.arange(interpolated_features.shape[1]), z_b_int].unsqueeze(-1)
                    features = torch.cat((r, g, b), dim=-1)
                
                return features
        
        if int_to_cano:
            interpolated_features = torch.zeros(self.batch_size, voxels_coo.size(1), features.size(1)//self.mapping_dim).to(features.device)
            if not rgb:
                if self.mapping_dim == features.shape[1]:
                    x_int = ((canonical_voxels_coo[..., 0]+1)/2 * (features.shape[3]-1)).long()
                    y_int = ((canonical_voxels_coo[..., 1]+1)/2 * (features.shape[2]-1)).long()
                    z_int = ((canonical_voxels_coo[..., 2]+1)/2 * (self.mapping_dim-1)).long()
                    interpolated_features = features[torch.arange(features.shape[0]).unsqueeze(1), z_int, y_int, x_int].unsqueeze(-1)
                else:
                    print("Error: the mapping dimension is not equal to the feature dimension")
            if rgb:
                if self.mapping_dim*3 == features.shape[1]:
                    x_int = ((canonical_voxels_coo[..., 0]+1)/2 * (features.shape[3]-1)).long()
                    y_int = ((canonical_voxels_coo[..., 1]+1)/2 * (features.shape[2]-1)).long()
                    z_int = ((canonical_voxels_coo[..., 2]+1)/2 * (self.mapping_dim-1)).long()
                    r = features[torch.arange(features.shape[0]).unsqueeze(1), z_int, y_int, x_int]
                    g = features[torch.arange(features.shape[0]).unsqueeze(1), z_int+self.mapping_dim, y_int, x_int]
                    b = features[torch.arange(features.shape[0]).unsqueeze(1), z_int+self.mapping_dim*2, y_int, x_int]
                    interpolated_features = torch.stack((r, g, b), dim=-1)
                else:
                    print("Error: the mapping dimension is not equal to the feature dimension")

        return interpolated_features
    
    def voxel_mapping(self, matrix_mapping):
        active_voxels_coo = torch.nonzero(matrix_mapping == 1).float()

        for i in range(active_voxels_coo.shape[-1]):
            dim_min = 0
            dim_max = matrix_mapping.shape[i] - 1
            active_voxels_coo[:, i] = 2 * (active_voxels_coo[:, i] - dim_min) / (dim_max - dim_min) - 1

        return active_voxels_coo
    
    def dynamical_tfs_from_canonical(self, points_coo, smpl_tfs, betas, gender, batch=False):
        if batch:
            verts_deformed = torch.zeros(self.batch_size, points_coo.shape[1], points_coo.shape[2]).cuda().float()
            for i, gen in enumerate(gender):   # Each frame in the batch needs to have the skinning weights loaded separately
                deformer = SMPLDeformer(betas=betas[i], gender=gen, K=7)
                weights = deformer.query_weights(points_coo[i])
                verts_deformed[i] = skinning(points_coo[i].unsqueeze(0), weights, smpl_tfs[i]).data[0]
        else:
            verts_deformed = torch.zeros(self.batch_size, points_coo.shape[0], points_coo.shape[1]).cuda().float()
            for i, gen in enumerate(gender):   # Each frame in the batch needs to have the skinning weights loaded separately
                deformer = SMPLDeformer(betas=betas[i], gender=gen, K=7)
                weights = deformer.query_weights(points_coo)
                verts_deformed[i] = skinning(points_coo.unsqueeze(0), weights, smpl_tfs[i]).data[0]
        return verts_deformed
    
    def render_values_at_rays(self, dynamical_voxels_coo, occupancy_field, rgb_field, original_image):
        '''Render image from occupancy field and rgb field.
        In its current state, this method splats the rgb values of the occupied voxels to the image plane and compares it to an interpolation
        of the original image rgb values.
        '''
        height, width = original_image.shape[2:]
        # n_training_rays = torch.tensor(self.n_training_rays).cuda()

        occupancy_map = (occupancy_field > self.occupancy_threshold).float() 
        if torch.count_nonzero(occupancy_map)==0:
            occupancy_map[:, 0, 0] = 1  # This is to avoid the case where there are no occupied voxels

        if self.opt['suppress_voxels_coo']:
            occupied_voxels_coo = dynamical_voxels_coo * occupancy_map
        else:
            occupied_voxels_coo = dynamical_voxels_coo
        if self.opt['suppress_voxels_ov']:
            occupied_voxels_ov = occupancy_field * occupancy_map   # Non occupied voxels have their opacity set to 0
        else:
            occupied_voxels_ov = occupancy_field

        occupied_voxels_world_coo = occupied_voxels_coo   # Transformed before 
        occupied_voxels_uv, exceeded_width_indices, exceeded_height_indices = get_uv(occupied_voxels_world_coo, self.cam_loc, self.intrinsics, self.pose)
        occupied_voxels_ov[exceeded_width_indices] = 0
        occupied_voxels_ov[exceeded_height_indices] = 0
        
        if self.opt['suppress_voxels_rgb']:
            occupied_voxels_rgb = rgb_field * occupancy_map   # Also the rgb values of the non occupied voxels are set to 0
        else:
            occupied_voxels_rgb = rgb_field        
        
        if self.visualize_stats:
            with torch.no_grad():
                self.visualize_voxels((occupied_voxels_coo * occupancy_map)[0], output_file='occupied_voxels.png', world=True)
                try:
                    occupied_canonical_voxels = occupancy_map[0].view(self.matrix_mapping.shape)
                    occupied_canonical_voxels_coo = self.voxel_mapping(occupied_canonical_voxels)
                    self.visualize_voxels(occupied_canonical_voxels_coo, output_file='occupied_canonical_voxels.png', world=True)
                except:
                    pass

        # Voxel splatting ----
        if self.opt['voxel_splatting']:
            original_rgb_values = torch.zeros(self.batch_size, occupied_voxels_coo.shape[1], 3, device=original_image.device, requires_grad=False)
            rendered_rgb_values = torch.zeros(self.batch_size, occupied_voxels_coo.shape[1], 3, device=original_image.device)
            
            # Convert to long and get fractional part
            x_int, y_int = occupied_voxels_uv[..., 0].long(), occupied_voxels_uv[..., 1].long()
            x_frac, y_frac = occupied_voxels_uv[..., 0] - x_int.float(), occupied_voxels_uv[..., 1] - y_int.float()

            batch_idx = torch.arange(self.batch_size).unsqueeze(1).to(y_int.device)  

            # -----------
            test_image = torch.zeros_like(original_image).cuda()
            test_image[batch_idx, :, y_int[0], x_int[0]] = torch.ones(3).cuda()
            wandb.log({"Test image": [wandb.Image(test_image[0].detach().cpu().numpy().transpose(1, 2, 0))]})
            wandb.log({"Original image": [wandb.Image(original_image[0].detach().cpu().numpy().transpose(1, 2, 0))]})
            save_image(test_image, 'outputs/stats/test_image.png')
            # -----------

            # Get the neighboring pixels
            top_left = original_image[batch_idx, :, y_int, x_int]
            top_right = original_image[batch_idx, :, y_int, torch.clamp(x_int + 1, 0, width - 1)]
            bottom_left = original_image[batch_idx, :, torch.clamp(y_int + 1, 0, height - 1), x_int]
            bottom_right = original_image[batch_idx, :, torch.clamp(y_int + 1, 0, height - 1),  torch.clamp(x_int + 1, 0, width - 1)]

            # Perform bilinear interpolation
            original_rgb_values = (top_left * ((1 - y_frac) * (1 - x_frac)).unsqueeze(-1).repeat(1,1,3) + \
                                top_right * ((1 - y_frac) * x_frac).unsqueeze(-1).repeat(1,1,3) + \
                                bottom_left * (y_frac * (1 - x_frac)).unsqueeze(-1).repeat(1,1,3) + \
                                bottom_right * (y_frac * x_frac).unsqueeze(-1).repeat(1,1,3))

            # Compute depth
            eps = 1e-6
            depth = torch.norm(occupied_voxels_world_coo - self.cam_loc.unsqueeze(1), dim=-1).unsqueeze(-1)
            depth = (depth - depth.min(dim=1, keepdim=True)[0]) / (depth.max(dim=1, keepdim=True)[0] - depth.min(dim=1, keepdim=True)[0])

            weights = torch.exp(2*occupied_voxels_ov) * torch.exp(1 - depth) / (torch.exp(1 - occupied_voxels_ov) + eps)
            #weights = (occupied_voxels_ov**2) * (1 - depth) / ((1 - occupied_voxels_ov)**2)
            weights = weights / (torch.sum(weights, dim=1, keepdim=True)[0] + eps)
            weights = (weights - weights.min(dim=1, keepdim=True)[0]) / (weights.max(dim=1, keepdim=True)[0] - weights.min(dim=1, keepdim=True)[0] + eps)
            
            weights_0 = weights[0, :10].detach().view(10, -1).cpu().numpy()
            wandb.log({"weights for first 10 rays histogram:": wandb.Histogram(weights_0)})

            # Compute rendered_rgb_values
            if self.opt['render_with_ov']:
                rendered_rgb_values = occupied_voxels_ov * occupied_voxels_rgb  
                #rendered_rgb_values = occupied_voxels_ov * (occupied_voxels_rgb.clone().detach())  
            else:
                rendered_rgb_values = occupied_voxels_rgb 
            #original_rgb_values = original_rgb_values * occupancy_map.clone().detach()   # Suppress rgb values corresponding to non occupied voxels
            #original_rgb_values = original_rgb_values + (rendered_rgb_values.clone().detach() - original_rgb_values) * (1 - weights.clone().detach())   # Correction term for further voxels
        # ----------
            '''
            original_rgb_values_add = torch.zeros(self.batch_size, occupied_voxels_coo.shape[1], 3, device=original_image.device)
            rendered_rgb_values_add = torch.zeros(self.batch_size, occupied_voxels_coo.shape[1], 3, device=original_image.device)

            original_rgb_values_add = original_rgb_values
            rendered_rgb_values_add = weights * original_rgb_values

            rendered_rgb_values = torch.cat((rendered_rgb_values, rendered_rgb_values_add), dim=1)
            original_rgb_values = torch.cat((original_rgb_values, original_rgb_values_add), dim=1)
            '''
        # ----------
        # Add randomly sampled rays to enforce constistency
        if self.opt['n_training_rays'] > 0:
            n_rays = self.opt['n_training_rays']
            original_rgb_values_add = torch.zeros(self.batch_size, n_rays, 3, device=original_image.device, requires_grad=False)
            rendered_rgb_values_add = torch.zeros(self.batch_size, n_rays, 3, device=original_image.device)

            if self.opt['train_on_non_black']:
                non_black_indices = torch.nonzero(original_image.sum(dim=1) > 0)
                indices_by_batch = [non_black_indices[non_black_indices[:, 0] == i] for i in range(original_image.shape[0])]
                selected_indices = [indices[torch.randperm(indices.shape[0])[:n_rays], 1:] for indices in indices_by_batch]
                selected_indices = torch.stack(selected_indices)[..., [1, 0]]
            else:
                height, width = original_image.shape[-2:]
                y_indices = torch.randint(height, (self.batch_size, n_rays), device='cuda')  # replace 'cuda' with your device
                x_indices = torch.randint(width, (self.batch_size, n_rays), device='cuda')  # replace 'cuda' with your device
                selected_indices = torch.stack((x_indices, y_indices), dim=-1)

            test, _ = get_rays(selected_indices, self.intrinsics, self.pose)
            # Visualize the rays ----------
            num_points = 100
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ray_dirs_0 = self.fov_bounds[0]
            test_0 = test[0]

            ax.scatter(occupied_voxels_world_coo[0, :, 0].detach().cpu(), occupied_voxels_world_coo[0, :, 1].detach().cpu(), occupied_voxels_world_coo[0, :, 2].detach().cpu(), c='k', marker='o', s=0.1)
            
            central_ray = F.normalize(torch.mean(ray_dirs_0, dim=0), dim=0)
            t = torch.linspace(0, 1.8, num_points).cuda()  # Change the range as needed
            points = (self.cam_loc[0] + central_ray * t[:, None] * (self.standard_depth_n+1)/2).detach()
            ax.scatter(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(), points[:, 2].cpu().numpy(), color='red', linewidth=0.1)

            body_center = self.smpl_params[:, 4:6].clone().unsqueeze(1)
            body_center[:,:,0] = (body_center[:,:,0]+1)/2*self.img_size[1]
            body_center[:,:,1] = (body_center[:,:,1]+1)/2*self.img_size[0]
            body_center_ray, _ = get_rays(body_center, self.intrinsics, self.pose)
            #body_center_ray = torch.normalize(body_center_ray, dim=2)
            body_center_ray_0 = body_center_ray[0][0]
            t = torch.linspace(0, 4, num_points).cuda()  # Change the range as needed
            points = (self.cam_loc[0] + body_center_ray_0 * t[:, None]).detach()
            ax.scatter(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(), points[:, 2].cpu().numpy(), color='green', linewidth=0.1)

            x_m_ray = torch.mean(ray_dirs_0[-2:], dim=0)
            y_m_ray = torch.mean(torch.stack((ray_dirs_0[0], ray_dirs_0[2]), dim=0), dim=0)
            t = torch.linspace(0, 4, num_points).cuda()  # Change the range as needed
            points = self.cam_loc[0] + x_m_ray * t[:, None]
            ax.scatter(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(), points[:, 2].cpu().numpy(), color='black', linewidth=0.1)
            points = self.cam_loc[0] + y_m_ray * t[:, None]
            ax.scatter(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(), points[:, 2].cpu().numpy(), color='black', linewidth=0.1)

            cluster_center = torch.tensor([0, 0, 0]).unsqueeze(0).unsqueeze(1).repeat(self.batch_size, 1, 1).cuda()
            cluster_center = self.norm_to_world_coo(cluster_center, self.cam_loc, self.fov_bounds).detach()
            cluster_center_uv, _, _ = get_uv(cluster_center, self.cam_loc, self.intrinsics, self.pose)
            ax.scatter(cluster_center[0, 0, 0].cpu().numpy(), cluster_center[0, 0, 1].cpu().numpy(), cluster_center[0, 0, 2].cpu().numpy(), color='red', linewidth=2)

            for i in range(min(16, test_0.shape[0])):
                dir = test_0[i]
                t = torch.linspace(0, 4, num_points).cuda()  # Change the range as needed
                points = self.cam_loc[0] + dir * t[:, None]

                ax.scatter(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(), points[:, 2].cpu().numpy(), linewidth=0.1, alpha=0.6)

            for i in range(ray_dirs_0.shape[0]):
                dir = ray_dirs_0[i]
                t = torch.linspace(0, 4, num_points).cuda()  # Change the range as needed
                points = self.cam_loc[0] + dir * t[:, None]

                ax.scatter(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(), points[:, 2].cpu().numpy(), color='black', linewidth=0.1)

            x, y, z = self.cam_loc[0].cpu().numpy() + torch.tensor([0, 0, 2]).cpu().numpy()
            text = f"scale: {self.scale[0].detach().cpu().numpy()} \ntrans: {self.trans[0].detach().cpu().numpy()}"  # replace with the text you want to display
            ax.text(x, y, z, text, color='red', fontsize=12)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            wandb.log({"Rays": [wandb.Image(fig)]})
            plt.close()
            # ------------------------------
            # Expand dimensions for broadcasting
            selected_indices_exp = selected_indices.unsqueeze(2)   # (batch, n_rays, 1, 2)
            occupied_voxels_uv_exp = occupied_voxels_uv.unsqueeze(1)   # (batch, 1, n_voxels, 2)  
            if self.opt['nearest_voxels'] == -1:
                # Calculate Euclidean distance
                distances = torch.norm(selected_indices_exp - occupied_voxels_uv_exp, dim=-1).unsqueeze(-1)   # (batch, n_rays, n_voxels)

                # Create a mask for distances less than the threshold
                threshold = self.opt['closeness_threshold']
                mask = (distances < threshold)
                close_voxels_number = torch.sum(mask, dim=2, keepdim=True)
                close_voxels_number_0 = close_voxels_number[0].detach().cpu().numpy()
                wandb.log({"Number of close voxels histogram:": wandb.Histogram(close_voxels_number_0)})

                # Apply the mask to get the distances and indices of the voxels within the threshold
                distances = distances / threshold 
                closest_voxel_distances = torch.where(mask, distances, torch.ones_like(distances))   # far voxels have distance 1
                #min_vals, _ = torch.min(closest_voxel_distances, dim=-1, keepdim=True)
                #max_vals, _ = torch.max(closest_voxel_distances, dim=-1, keepdim=True)
                #eps = 1e-6
                #closest_voxel_distances = ((closest_voxel_distances - min_vals) / (max_vals - min_vals + eps))

                # Expand occupied_voxels_ov to match the shape of mask
                occupied_voxels_ov_expanded = occupied_voxels_ov.unsqueeze(1).expand(-1, mask.shape[1], -1, -1)   # Opacity 0 for far voxels
                occupied_voxels_rgb_expanded = occupied_voxels_rgb.unsqueeze(1).expand(-1, mask.shape[1], -1, -1)   # RGB 0 for far voxels

                # Apply the mask to the tensors
                occupied_voxels_ov_selected = torch.where(mask, occupied_voxels_ov_expanded, torch.zeros_like(occupied_voxels_ov_expanded))   # Put ones for the formula, it nullifies non considered voxels' contributes
                occupied_voxels_rgb_selected = torch.where(mask, occupied_voxels_rgb_expanded, torch.zeros_like(occupied_voxels_rgb_expanded))
                #occupied_voxels_ov_selected = occupied_voxels_ov_expanded * mask
                #occupied_voxels_rgb_selected = occupied_voxels_rgb_expanded * mask

                eps = 1e-6
                depth_all = torch.norm(occupied_voxels_world_coo - self.cam_loc.unsqueeze(1), dim=-1)
                depth_all = (depth_all - depth_all.min(dim=1, keepdim=True)[0]) / (depth_all.max(dim=1, keepdim=True)[0] - depth_all.min(dim=1, keepdim=True)[0])
                depth_all = depth_all.unsqueeze(1).unsqueeze(-1).expand_as(mask)
                depth = torch.where(mask, depth_all, torch.zeros_like(depth_all))   # far voxels are suppress to normalize
                depth = (depth - depth.min(dim=2, keepdim=True)[0]) / (depth.max(dim=2, keepdim=True)[0] - depth.min(dim=2, keepdim=True)[0] + eps)
                depth = torch.where(mask, depth, torch.ones_like(depth))   # far voxels have distance 1

                weights = torch.exp(occupied_voxels_ov_selected) * torch.exp(1 - depth)
                #weights = (occupied_voxels_ov_selected**2) * (1 - depth) / ((1 - occupied_voxels_ov_selected)**2)
                weights = weights / (torch.sum(torch.where(mask, weights, torch.zeros_like(weights)), dim=2, keepdim=True)[0] + eps)
                weights = (1 + weights - weights.min(dim=2, keepdim=True)[0]) / (1 + weights.max(dim=2, keepdim=True)[0] - weights.min(dim=2, keepdim=True)[0] + eps)
                weights_0 = weights[0, :10].detach().view(10, -1).cpu().numpy()
                wandb.log({"weights for first 10 rays histogram:": wandb.Histogram(weights_0)})

                eps = 1e-6
                # Compute rendered_rgb_values_add
                if self.opt['render_with_ov']:
                    exp = 1
                    #rendered_rgb_values_add = torch.sum(occupied_voxels_rgb_selected * ((1 - closest_voxel_distances)**exp), dim=2) / (torch.sum((1 - closest_voxel_distances)**exp, dim=2) + eps)
                    rendered_rgb_values_add = torch.sum(weights * occupied_voxels_rgb_selected * ((1 - closest_voxel_distances)**exp), dim=2) / (torch.sum(weights, dim=2) + torch.sum((1 - closest_voxel_distances)**exp, dim=2) + eps)
                else:
                    rendered_rgb_values_add = torch.sum(occupied_voxels_rgb_selected * ((1 - closest_voxel_distances)**exp) * ((1 - depth)**exp), dim=2) / (torch.sum((1 - depth)**exp, dim=2) + torch.sum((1-closest_voxel_distances)**exp, dim=2) + eps)

            u, v = selected_indices[..., 0], selected_indices[..., 1]
            batch_idx = torch.arange(self.batch_size).unsqueeze(1).to(dynamical_voxels_coo.device)
            original_rgb_values_add = original_image[batch_idx, :, v, u]
            
            # Concatenate along the second dimension
            if self.opt['voxel_splatting']:
                rendered_rgb_values = torch.cat((rendered_rgb_values, rendered_rgb_values_add), dim=1)
                original_rgb_values = torch.cat((original_rgb_values, original_rgb_values_add), dim=1)
            else:
                rendered_rgb_values = rendered_rgb_values_add
                original_rgb_values = original_rgb_values_add
        
        '''This is working but is pretty slow, 4s/image for 16x16x16 voxels
        for image_idx in range(original_image.shape[0]):
            depth = torch.norm(occupied_voxels_world_coo[image_idx, :, :] - self.cam_loc[image_idx].unsqueeze(0).repeat(occupied_voxels_world_coo.shape[1], 1), dim=-1).cuda()
            depth = (depth - depth.min()) / (depth.max() - depth.min()) 
            for voxel_idx in range(occupied_voxels_uv.shape[1]):
                x, y = occupied_voxels_uv[image_idx, voxel_idx, 0], occupied_voxels_uv[image_idx, voxel_idx, 1]
                y_int, x_int = y.long().item(), x.long().item()
                y_frac, x_frac = y - y_int, x - x_int
                 
                original_rgb_values[image_idx, voxel_idx, :] = original_image[image_idx, :, y_int, x_int] * (y_frac * x_frac) + \
                                                        original_image[image_idx, :, y_int, x_int+1] * (y_frac * (1-x_frac)) + \
                                                        original_image[image_idx, :, y_int+1, x_int] * ((1-y_frac) * x_frac) + \
                                                        original_image[image_idx, :, y_int+1, x_int+1] * ((1-y_frac) * (1-x_frac))
                
                rendered_rgb_values[image_idx, voxel_idx, :] = occupied_voxels_ov[image_idx, voxel_idx] * occupied_voxels_rgb[image_idx, voxel_idx, :] * (1 - depth[voxel_idx])/torch.exp(torch.tensor(1))
        '''

        if self.visualize_stats:
            with torch.no_grad():
                rgb_values_differences = original_rgb_values - rendered_rgb_values
                strings_height = 64
                try:
                    original_values = original_rgb_values.view(self.batch_size, 3, strings_height, -1)
                    rendered_values = rendered_rgb_values.view(self.batch_size, 3, strings_height, -1)
                    differences_values = rgb_values_differences.view(self.batch_size, 3, strings_height, -1)
                except:
                    original_values = original_rgb_values.view(self.batch_size, 3, 1, -1)
                    rendered_values = rendered_rgb_values.view(self.batch_size, 3, 1, -1)
                    differences_values = rgb_values_differences.view(self.batch_size, 3, 1, -1)
                images = torch.cat((original_values[:, :, :, :], rendered_values[:, :, :, :], differences_values[:, :, :, :]))
                grid = make_grid(images, nrow=self.batch_size)  # Arrange the images in a 3xbtach_size grid
                wandb.log({'original_vs_rendered_rgb_values': [wandb.Image(grid)]})
                
        return original_rgb_values, rendered_rgb_values
    
    def render_image(self, dynamical_voxels_coo, occupancy_field, rgb_field,  original_image):
        '''Render image from occupancy field and rgb field.'''
        height, width = original_image.shape[2:]
        # n_training_rays = torch.tensor(self.n_training_rays).cuda()

        occupancy_map = (occupancy_field > self.occupancy_threshold).float() 
        if torch.count_nonzero(occupancy_map)==0:
            occupancy_map[:, 0, 0] = 1  # This is to avoid the case where there are no occupied voxels

        if self.opt['suppress_voxels_coo']:
            occupied_voxels_coo = dynamical_voxels_coo * occupancy_map
        else:
            occupied_voxels_coo = dynamical_voxels_coo
        if self.opt['suppress_voxels_ov']:
            occupied_voxels_ov = occupancy_field * occupancy_map   # Non occupied voxels have their opacity set to 0
        else:
            occupied_voxels_ov = occupancy_field

        occupied_voxels_world_coo = occupied_voxels_coo
        occupied_voxels_uv, exceeded_width_indices, exceeded_height_indices = get_uv(occupied_voxels_world_coo, self.cam_loc, self.intrinsics, self.pose)
        occupied_voxels_ov[exceeded_width_indices] = 0
        occupied_voxels_ov[exceeded_height_indices] = 0
        
        if self.opt['suppress_voxels_rgb']:
            occupied_voxels_rgb = rgb_field * occupancy_map   # Also the rgb values of the non occupied voxels are set to 0
        else:
            occupied_voxels_rgb = rgb_field
        
        n_rays = height*width
        rendered_rgb_values = torch.zeros(self.batch_size, n_rays, 3, device=original_image.device)

        height, width = original_image.shape[-2:]
        y_indices, x_indices = torch.meshgrid(torch.arange(height), torch.arange(width))
        y_indices = y_indices.flatten().to(device='cuda')
        x_indices = x_indices.flatten().to(device='cuda')
        pixel_indices = torch.stack((x_indices, y_indices), dim=-1).unsqueeze(0).repeat(self.batch_size, 1, 1)
        print(f'Pixel indices shape: {pixel_indices.shape}')
        print("Pixel indices:", pixel_indices[0, :1920])
        
        n_patches = 1000
        n_pixels = n_rays // n_patches
        for i in range(n_patches):
            print(f'Processing patch {i+1}/{n_patches}', end='\r')
            selected_indices = pixel_indices[:, i*n_pixels:(i+1)*n_pixels, :]

            selected_indices_exp = selected_indices.unsqueeze(2)   # (batch, n_rays, 1, 2)
            occupied_voxels_uv_exp = occupied_voxels_uv.unsqueeze(1)   # (batch, 1, n_voxels, 2)  
            if self.opt['render_with_rays']:   
                # Calculate Euclidean distance
                distances = torch.norm(selected_indices_exp - occupied_voxels_uv_exp, dim=-1).unsqueeze(-1)   # (batch, n_rays, n_voxels)

                # Create a mask for distances less than the threshold
                threshold = self.opt['closeness_threshold']
                mask = (distances < threshold)

                # Apply the mask to get the distances and indices of the voxels within the threshold
                distances = distances / threshold 
                closest_voxel_distances = torch.where(mask, distances, torch.ones_like(distances))   # far voxels have distance 1
                #min_vals, _ = torch.min(closest_voxel_distances, dim=-1, keepdim=True)
                #max_vals, _ = torch.max(closest_voxel_distances, dim=-1, keepdim=True)
                #eps = 1e-6
                #closest_voxel_distances = ((closest_voxel_distances - min_vals) / (max_vals - min_vals + eps))

                # Expand occupied_voxels_ov to match the shape of mask
                occupied_voxels_ov_expanded = occupied_voxels_ov.unsqueeze(1).expand(-1, mask.shape[1], -1, -1)   # Opacity 0 for far voxels
                occupied_voxels_rgb_expanded = occupied_voxels_rgb.unsqueeze(1).expand(-1, mask.shape[1], -1, -1)   # RGB 0 for far voxels

                # Apply the mask to the tensors
                occupied_voxels_ov_selected = torch.where(mask, occupied_voxels_ov_expanded, torch.zeros_like(occupied_voxels_ov_expanded))   # Put ones for the formula, it nullifies non considered voxels' contributes
                occupied_voxels_rgb_selected = torch.where(mask, occupied_voxels_rgb_expanded, torch.zeros_like(occupied_voxels_rgb_expanded))
                #occupied_voxels_ov_selected = occupied_voxels_ov_expanded * mask
                #occupied_voxels_rgb_selected = occupied_voxels_rgb_expanded * mask

                eps = 1e-6
                depth_all = torch.norm(occupied_voxels_world_coo - self.cam_loc.unsqueeze(1), dim=-1)
                depth_all = (depth_all - depth_all.min(dim=1, keepdim=True)[0]) / (depth_all.max(dim=1, keepdim=True)[0] - depth_all.min(dim=1, keepdim=True)[0])
                depth_all = depth_all.unsqueeze(1).unsqueeze(-1).expand_as(mask)
                depth = torch.where(mask, depth_all, torch.zeros_like(depth_all))   # far voxels are suppress to normalize
                depth = (depth - depth.min(dim=2, keepdim=True)[0]) / (depth.max(dim=2, keepdim=True)[0] - depth.min(dim=2, keepdim=True)[0] + eps)
                depth = torch.where(mask, depth, torch.ones_like(depth))   # far voxels have distance 1

                weights = torch.exp(occupied_voxels_ov_selected) * torch.exp(1 - depth)
                #weights = (occupied_voxels_ov_selected**2) * (1 - depth) / ((1 - occupied_voxels_ov_selected)**2)
                weights = weights / (torch.sum(torch.where(mask, weights, torch.zeros_like(weights)), dim=2, keepdim=True)[0] + eps)
                weights = (weights - weights.min(dim=2, keepdim=True)[0]) / (weights.max(dim=2, keepdim=True)[0] - weights.min(dim=2, keepdim=True)[0] + eps)
                weights_0 = weights[0, :10].detach().view(10, -1).cpu().numpy()
                wandb.log({"weights for first 10 rays histogram:": wandb.Histogram(weights_0)})

                eps = 1e-6
                # Compute rendered_rgb_values_add
                if self.opt['render_with_ov']:
                    exp = 1
                    rendered_rgb_values[:, i*n_pixels:(i+1)*n_pixels, :] = torch.sum(weights * occupied_voxels_rgb_selected * ((1 - closest_voxel_distances)**exp), dim=2) / (torch.sum(weights, dim=2) + torch.sum((1 - closest_voxel_distances)**exp, dim=2) + eps)
                    #rendered_rgb_values_add = torch.sum(occupied_voxels_ov_selected * occupied_voxels_rgb_selected * (1 - closest_voxel_distances) * (1 - depth), dim=2) / (torch.sum(occupied_voxels_ov_selected, dim=2) + torch.sum((1 - depth), dim=2) + torch.sum((1 - closest_voxel_distances), dim=2) + eps)
                else:
                    rendered_rgb_values[:, i*n_pixels:(i+1)*n_pixels, :] = torch.sum(occupied_voxels_rgb_selected * ((1 - closest_voxel_distances)**exp) * ((1 - depth)**exp), dim=2) / (torch.sum((1 - depth)**exp, dim=2) + torch.sum((1-closest_voxel_distances)**exp, dim=2) + eps)

        rendered_image = rendered_rgb_values.view(self.batch_size, 3, height, width)

        return rendered_image
    
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
