from typing import Union, NamedTuple
import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import math



class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

class CustomPad(nn.Module):
    def __init__(self, padding):
      self.padding = padding
    
    def forward(self, input: torch.Tensor):
      return F.pad(input, self.padding)


class UNET(nn.Module):
    def __init__(self, classes:int ,height:int, width:int, channels:int):
        super().__init__()

        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.classes = classes

        self.l1conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels = 64,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.initialise_layer(self.l1conv1)
        self.l1conv2 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.initialise_layer(self.l1conv2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))

        self.l2conv1 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.initialise_layer(self.l2conv1)
        self.l2conv2 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.initialise_layer(self.l2conv2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))

        self.l3conv1 = nn.Conv2d(
            in_channels = 128,
            out_channels = 256,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.initialise_layer(self.l3conv1)
        self.l3conv2 = nn.Conv2d(
            in_channels = 256,
            out_channels = 256,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.initialise_layer(self.l3conv2)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))

        self.l4conv1 = nn.Conv2d(
            in_channels = 256,
            out_channels = 512,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.initialise_layer(self.l4conv1)
        self.l4conv2 = nn.Conv2d(
            in_channels = 512,
            out_channels = 512,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.initialise_layer(self.l4conv2)
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2))

        self.l5conv1 = nn.Conv2d(
            in_channels = 512,
            out_channels = 1024,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.initialise_layer(self.l5conv1)
        self.l5conv2 = nn.Conv2d(
            in_channels = 1024,
            out_channels = 1024,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.initialise_layer(self.l5conv2)

        self.dropout1 = nn.Dropout2d(p=0.5)
        self.dropout2 = nn.Dropout2d(p=0.5)

        #expanding path

        self.up1 = nn.Upsample(scale_factor=(2,2))
        self.l6conv1 = nn.Conv2d(
            in_channels = 1024,
            out_channels = 512,
            kernel_size = (2,2),
        )
        self.initialise_layer(self.l6conv1)
        self.pad6 = CustomPad((0,1,0,1)).forward
        self.l6conv2 = nn.Conv2d(
            in_channels = 512,
            out_channels = 512,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.initialise_layer(self.l6conv2)

        self.l6conv3 = nn.Conv2d(
            in_channels = 512,
            out_channels = 512,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.initialise_layer(self.l6conv3)

        self.up2 = nn.Upsample(scale_factor=(2,2))
        self.l7conv1 = nn.Conv2d(
            in_channels = 512,
            out_channels = 256,
            kernel_size = (2,2),
        )
        self.initialise_layer(self.l7conv1)
        self.pad7 = CustomPad((0,1,0,1)).forward
        self.l7conv2 = nn.Conv2d(
            in_channels = 256,
            out_channels = 256,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.initialise_layer(self.l7conv2)

        self.l7conv3 = nn.Conv2d(
            in_channels = 256,
            out_channels = 256,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.initialise_layer(self.l7conv3)

        self.up3 = nn.Upsample(scale_factor=(2,2))
        self.l8conv1 = nn.Conv2d(
            in_channels = 256,
            out_channels = 128,
            kernel_size = (2,2),
        )
        self.initialise_layer(self.l8conv1)
        self.pad8 = CustomPad((0,1,0,1)).forward
        self.l8conv2 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.initialise_layer(self.l8conv2)

        self.l8conv3 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.initialise_layer(self.l8conv3)

        self.up4 = nn.Upsample(scale_factor=(2,2))
        self.l9conv1 = nn.Conv2d(
            in_channels = 128,
            out_channels = 64,
            kernel_size = (2,2),
        )
        self.initialise_layer(self.l9conv1)
        self.pad9 = CustomPad((0,1,0,1)).forward
        self.l9conv2 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.initialise_layer(self.l9conv2)

        self.l9conv3 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.initialise_layer(self.l9conv3)
        self.l9conv4 = nn.Conv2d(
            in_channels = 64,
            out_channels = self.classes,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.initialise_layer(self.l9conv4)

        self.l10conv1 = nn.Conv2d(
            in_channels = self.classes,
            out_channels = 1,
            kernel_size = (1,1)
        )
        self.initialise_layer(self.l10conv1)

    def forward(self, images: torch.Tensor):
        #contracting path
        c11_out = F.relu(self.l1conv1(images))
        c12_out = F.relu(self.l1conv2(c11_out))
        p1_out = self.pool1(c12_out)

        c21_out = F.relu(self.l2conv1(p1_out))
        c22_out = F.relu(self.l2conv2(c21_out))
        p2_out = self.pool2(c22_out)

        c31_out = F.relu(self.l3conv1(p2_out))
        c32_out = F.relu(self.l3conv2(c31_out))
        p3_out = self.pool3(c32_out)

        c41_out = F.relu(self.l4conv1(p3_out))
        c42_out = F.relu(self.l4conv2(c41_out))
        c42_out = self.dropout1(c42_out)
        p4_out = self.pool4(c42_out)

        c51_out = F.relu(self.l5conv1(p4_out))
        c52_out = F.relu(self.l5conv2(c51_out))
        drop_out = self.dropout2(c52_out)

        #Expanding path
        up1_out = self.up1(drop_out)
        c61_out = F.relu(self.pad6(self.l6conv1(up1_out)))
        c62_out = F.relu(self.l6conv2(c61_out))
        c63_out = F.relu(self.l6conv3(c62_out))

        up2_out = self.up2(c63_out)
        c71_out = F.relu(self.pad7(self.l7conv1(up2_out)))
        c72_out = F.relu(self.l7conv2(c71_out))
        c73_out = F.relu(self.l7conv3(c72_out))

        up3_out = self.up3(c73_out)
        c81_out = F.relu(self.pad8(self.l8conv1(up3_out)))
        c82_out = F.relu(self.l8conv2(c81_out))
        c83_out = F.relu(self.l8conv3(c82_out))

        up4_out = self.up4(c83_out)
        c91_out = F.relu(self.pad9(self.l9conv1(up4_out)))
        c92_out = F.relu(self.l9conv2(c91_out))
        c93_out = F.relu(self.l9conv3(c92_out))
        c94_out = F.relu(self.l9conv4(c93_out))

        c10_out = self.l10conv1(c94_out)
        
        return c10_out


class UNET2(nn.Module):
    def __init__(self, classes:int ,height:int, width:int, channels:int):
        super().__init__()

        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.classes = classes

        self.pad = nn.ZeroPad2d((0,1,0,1))

        self.l1conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels = 64,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.l1conv2 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))

        self.l2conv1 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.l2conv2 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))

        self.l3conv1 = nn.Conv2d(
            in_channels = 128,
            out_channels = 256,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.l3conv2 = nn.Conv2d(
            in_channels = 256,
            out_channels = 256,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))

        self.l4conv1 = nn.Conv2d(
            in_channels = 256,
            out_channels = 512,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.l4conv2 = nn.Conv2d(
            in_channels = 512,
            out_channels = 512,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2))

        self.l5conv1 = nn.Conv2d(
            in_channels = 512,
            out_channels = 1024,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.l5conv2 = nn.Conv2d(
            in_channels = 1024,
            out_channels = 1024,
            kernel_size = (3,3),
            padding = (1,1)
        )

        self.dropout1 = nn.Dropout(p=0.5, inplace=True)
        self.dropout2 = nn.Dropout(p=0.5, inplace=True)

        #expanding path

        self.up1 = nn.Upsample(scale_factor=2)
        self.l6conv1 = nn.Conv2d(
            in_channels = 1024,
            out_channels = 512,
            kernel_size = (2,2),
        )
        self.l6conv2 = nn.Conv2d(
            in_channels = 1024,
            out_channels = 512,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.l6conv3 = nn.Conv2d(
            in_channels = 512,
            out_channels = 512,
            kernel_size = (3,3),
            padding = (1,1)
        )

        self.up2 = nn.Upsample(scale_factor=2)
        self.l7conv1 = nn.Conv2d(
            in_channels = 512,
            out_channels = 256,
            kernel_size = (2,2),
        )
        self.l7conv2 = nn.Conv2d(
            in_channels = 512,
            out_channels = 256,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.l7conv3 = nn.Conv2d(
            in_channels = 256,
            out_channels = 256,
            kernel_size = (3,3),
            padding = (1,1)
        )

        self.up3 = nn.Upsample(scale_factor=2)
        self.l8conv1 = nn.Conv2d(
            in_channels = 256,
            out_channels = 128,
            kernel_size = (2,2),
        )
        self.l8conv2 = nn.Conv2d(
            in_channels = 256,
            out_channels = 128,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.l8conv3 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = (3,3),
            padding = (1,1)
        )

        self.up4 = nn.Upsample(scale_factor=2)
        self.l9conv1 = nn.Conv2d(
            in_channels = 128,
            out_channels = 64,
            kernel_size = (2,2),
        )
        self.l9conv2 = nn.Conv2d(
            in_channels = 128,
            out_channels = 64,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.l9conv3 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.l9conv4 = nn.Conv2d(
            in_channels = 64,
            out_channels = self.classes,
            kernel_size = (1,1),
        )
        # self.l10conv1 = nn.Conv2d(
        #     in_channels = self.classes,
        #     out_channels = 1,
        #     kernel_size = (1,1)
        # )

    def forward(self, images: torch.Tensor):
        #contracting path
        conv1 = F.relu(self.l1conv1(images))
        conv1 = F.relu(self.l1conv2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = F.relu(self.l2conv1(pool1))
        conv2 = F.relu(self.l2conv2(conv2))
        pool2 = self.pool2(conv2)

        conv3 = F.relu(self.l3conv1(pool2))
        conv3 = F.relu(self.l3conv2(conv3))
        pool3 = self.pool3(conv3)

        conv4 = F.relu(self.l4conv1(pool3))
        conv4 = F.relu(self.l4conv2(conv4))
        drop4 = self.dropout1(conv4)
        pool4 = self.pool4(conv4)

        conv5 = F.relu(self.l5conv1(pool4))
        conv5 = F.relu(self.l5conv2(conv5))
        drop5 = self.dropout2(conv5)

        #Expanding path
        up6 = self.up1(conv5)
        up6 = self.pad(F.relu(self.l6conv1(up6)))
        up6 = torch.cat((drop4, up6), dim = 1)
        conv6 = F.relu(self.l6conv2(up6))
        conv6 = F.relu(self.l6conv3(up6))

        up7 = self.up2(conv6)
        up7 = self.pad(F.relu((self.l7conv1(up7))))
        up7 = torch.cat((conv3, up7), dim=1)
        conv7 = F.relu(self.l7conv2(up7))
        conv7 = F.relu(self.l7conv3(up7))

        up8 = self.up3(conv7)
        up8 = self.pad(F.relu(self.l8conv1(up8)))
        up8 = torch.cat((conv2, up8), dim=1)
        conv8 = F.relu(self.l8conv2(up8))
        conv8 = F.relu(self.l8conv3(up8))

        up9 = self.up4(conv8)
        up9 = self.pad(F.relu(self.l9conv1(up9)))
        up9 = torch.cat((conv1, up9), dim=1)
        conv9 = F.relu(self.l9conv2(up9))
        conv9 = F.relu(self.l9conv3(up9))
        conv9 = F.relu(self.l9conv4(conv9))
        
        return self.l9conv4(conv9)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    
    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.relu(self.conv2(t))
        return t

class unet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_block1 = Block(1, 64)
        self.down_block2 = Block(64, 128)
        self.down_block3 = Block(128, 256)
        self.down_block4 = Block(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        self.bottle = Block(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.up_block1 = Block(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up_block2 = Block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_block3 = Block(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up_block4 = Block(128, 64)

        self.out = nn.Conv2d(64, 2, 1)

    def forward(self, t):
        down1 = self.down_block1(t)
        t = self.maxpool(down1)

        down2 = self.down_block2(t)
        t = self.maxpool(down2)

        down3 = self.down_block3(t)
        t = self.maxpool(down3)

        down4 = self.down_block4(t)
        t = self.maxpool(down4)
        t = F.dropout(t)

        t = self.bottle(t)
        t = F.dropout(t)

        t = self.up1(t)
        #t = torch.cat([t, self.crop(t, down4)], 1)
        t = self.up_block1(t)
        
        t = self.up2(t)
       # t = torch.cat([t, self.crop(t, down3)], 1)
        t = self.up_block2(t)

        t = self.up3(t)
        t = torch.cat([t, self.crop(t, down2)], 1)
        t = self.up_block3(t)

        t = self.up4(t)
        t = torch.cat([t, self.crop(t, down1)], 1)
        t = self.up_block4(t)

        t = self.out(t)
        return t
    
    @staticmethod
    def crop(t, d):
        pad_x1 = int(math.floor((t.shape[3] - d.shape[3]) / 2))
        pad_x2 = int(math.ceil((t.shape[3] - d.shape[3]) / 2))
        pad_y1 = int(math.floor((t.shape[2] - d.shape[2]) / 2))
        pad_y2 = int(math.ceil((t.shape[2] - d.shape[2]) / 2))
        return F.pad(d, (pad_x1, pad_x2, pad_y1, pad_y2))



    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.constant_(layer.bias, 0.1)
        if hasattr(layer, "weight"):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv1 = nn.Conv2d(in_channels, in_channels//2, 2)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv1(x1)
        # print(x1.size())
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.ZeroPad2d((diffX // 2,diffX - diffX // 2,diffY // 2,diffY - diffY // 2))(x1)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        #factor = 1 if bilinear else 1
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.p = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.outc = OutConv(2, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        drop4 = nn.Dropout()(x4)
        x5 = self.down4(drop4)
        drop5 = nn.Dropout()(x5)
        x = self.up1(drop5, drop4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.p(x)
        logits = self.outc(x)
        return logits






