import torch
import torch.nn as nn
from torchvision import transforms, models
from collections import OrderedDict
from lib.utils import DoubleConv, DownSample, UpSample, PixelShuffle3d, PixelUnshuffle3d


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
    