import torch
import torch.nn as nn


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
    
class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)

class PixelUnshuffle3d(nn.Module):
    '''
    This class is a 3D version of pixel unshuffle, the inverse operation of PixelShuffle3d.
    '''
    def __init__(self, scale):
        '''
        :param scale: downsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()

        # Ensure that the depth, height, and width are divisible by the scale factor
        assert in_depth % self.scale == 0, f"Incompatible depth {in_depth} for scale {self.scale}"
        assert in_height % self.scale == 0, f"Incompatible height {in_height} for scale {self.scale}"
        assert in_width % self.scale == 0, f"Incompatible width {in_width} for scale {self.scale}"

        out_depth = in_depth // self.scale
        out_height = in_height // self.scale
        out_width = in_width // self.scale
        nOut = channels * self.scale ** 3

        # Reshape the input to merge the spatial dimensions into the channel dimension
        input_view = input.view(batch_size, channels, out_depth, self.scale, out_height, self.scale, out_width, self.scale)
        output = input_view.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)