import torch
import torch.nn as nn
from lib.modules.modules import downsample_layer, upsample_layer


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
