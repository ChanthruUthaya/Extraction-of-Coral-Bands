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

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

class CustomPad(nn.Module):
    def __init__(self, padding):
      self.padding = padding
    
    def forward(self, input: torch.Tensor):
      return F.pad(input, self.padding)

class DoubConv(nn.Module):
    def __init__(self, in_channels, out_channels,init_data ,mid_channels=None):
            super().__init__()
            self.init_data = init_data
            if not mid_channels:
                mid_channels = out_channels
            self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
            initialise_layer(self.conv1, in_channels, init_data)
            self.bnorm1 = nn.BatchNorm2d(mid_channels)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
            initialise_layer(self.conv2, mid_channels, 9)
            self.bnorm2 = nn.BatchNorm2d(out_channels)
            self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bnorm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bnorm2(out)

        return self.relu2(out)

class DownLayer(nn.Module):
    def __init__(self, in_channels, out_channels, init_data):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubConv(in_channels, out_channels, init_data=init_data)
    
    def forward(self, x):
        out = self.maxpool(x)
        return self.conv(out)

class UpLayer(nn.Module):

    def __init__(self, in_channels, out_channels, pad, init_data):
        super().__init__()
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 2)
        initialise_layer(self.conv1, in_channels, init_data)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.padding = pad
        self.doubconv = DoubConv(out_channels, out_channels, init_data=4)
    
    def forward(self, x1):
        x1 = self.up1(x1)
        x1 = self.conv1(x1)
        x1 = self.bnorm1(x1)
        x1 = self.relu(x1)
        x1 = F.pad(x1, self.padding)
        return self.doubconv(x1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, classes, init_data):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels = in_channels,
            out_channels = classes,
            kernel_size = (3,3),
            padding = (1,1))
        self.bnorm = nn.BatchNorm2d(classes)
        self.relu = nn.ReLU(inplace=True)

        initialise_layer(self.conv, in_channels, init_data)

        self.conv2 = nn.Conv2d(classes, out_channels, kernel_size = 1)
        initialise_layer(self.conv2, classes, init_data)



    
    def forward(self, x):
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.relu(x)

        return self.conv2(x)
    

class UNET2D(nn.Module):
    def __init__(self, classes:int, height:int, width:int, channels:int):
        super().__init__()

        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.classes = classes

        self.batch_norm = nn.BatchNorm2d(self.input_shape.channels)

        self.down1 = DoubConv(self.input_shape.channels, 64, init_data=1)
        self.down2 = DownLayer(64, 128, init_data=9)
        self.down3 = DownLayer(128, 256, init_data=9)
        self.down4 = DownLayer(256,512, init_data=9)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.down5 = DownLayer(512,1024, init_data=9)
        self.dropout2 = nn.Dropout2d(p=0.5)
        #self.lstm = LSTMLayer()
        self.up6 = UpLayer(1024, 512, (0,1,0,1), init_data=9)
        self.up7 = UpLayer(512, 256, (0,1,0,1), init_data=9)
        self.up8 = UpLayer(256, 128, (0,1,0,1), init_data=9)
        self.up9 = UpLayer(128, 64, (0,1,0,1), init_data=9)

        # 
        # nn.Sequential(
        #             nn.Conv2d(in_channels = 64,
        #              out_channels = self.classes,
        #              kernel_size = (3,3),
        #              padding = (1,1)),
        #             nn.BatchNorm2d(self.classes),
        #             nn.ReLU(inplace=True)
        # )
        self.out10 = OutConv(64, 1,2, init_data=9)

    def forward(self, images: torch.Tensor):
        bout = self.batch_norm(images)

        out1 = self.down1(bout)
        out2 = self.down2(out1)
        out3 = self.down3(out2)

        out4 = self.down4(out3)
        dout4 = self.dropout1(out4)
    

        out5 = self.down5(dout4)
        dout5 = self.dropout2(out5)

        out6 = self.up6(dout5)
        out7 = self.up7(out6)
        out8 = self.up8(out7)
        out9 = self.up9(out8)

        out = self.out10(out9)


        return out

def initialise_layer(layer, prev_c, prev_k):
    if hasattr(layer, "bias"):
        nn.init.constant_(layer.bias, 0.1)
    if hasattr(layer, "weight"):
        n = prev_k*prev_c
        std = math.sqrt(2/n)
        nn.init.normal_(layer.weight, mean=0.0, std=std)


        

        




