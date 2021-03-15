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

from clstm import *

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
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.double_conv(x)

class DownLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxp_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubConv(in_channels, out_channels),
        )
    
    def forward(self, x):
        return self.maxp_conv(x)

class UpLayer(nn.Module):

    def __init__(self, in_channels, out_channels, pad):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size = 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.padding = pad
        self.doubconv = DoubConv(out_channels, out_channels)
    
    def forward(self, x1):
        x1 = self.initial(x1)
        x1 = F.pad(x1, self.padding)
        return self.doubconv(x1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
    
    def forward(self, x):
        return self.conv(x)

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
    

    def forward(self, images):
         # Squash samples and timesteps into a single axis
        batch, timesteps, channels, height, width = images.size()
        
        x_reshape = images.contiguous().view(batch*timesteps, channels, height, width)  # (batch*timestep, channels, height, width)

        y = self.module(x_reshape)
        _, channels, height, width = y.size()
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(batch, timesteps, channels, height, width)  # (samples, timesteps, output_size)
        else:
            y = y.view(timesteps, batch, channels, height, width)  # (timesteps, samples, output_size)

        return y
    
    
class UNETDown(nn.Module):
    def __init__(self, channels:int):
        super().__init__()

        #self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.channels = channels

        self.down1 = DoubConv(self.channels, 64)
        self.down2 = DownLayer(64, 128)
        self.down3 = DownLayer(128, 256)
        self.down4 = DownLayer(256,512)
        self.dropout1 = nn.Dropout2d(p=0.5)
        # self.down5 = DownLayer(512,1024)
        # self.dropout2 = nn.Dropout2d(p=0.5)
    
    def forward(self, images: torch.Tensor):
        out1 = self.down1(images)
        out2 = self.down2(out1)
        out3 = self.down3(out2)

        out4 = self.down4(out3)
        out4 = self.dropout1(out4)
    

        # out5 = self.down5(out4)
        # out5 = self.dropout2(out5)

        return out4

class UNETUp(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes

        self.maxout = nn.MaxPool2d(kernel_size=2)

        self.up6 = UpLayer(1024, 512, (0,1,0,1))
        self.up7 = UpLayer(512, 256, (0,1,0,1))
        self.up8 = UpLayer(256, 128, (0,1,0,1))
        self.up9 = UpLayer(128, 64, (0,1,0,1))
        
    def forward(self, images:torch.Tensor):

        out5 = self.maxout(images)

        print("LSTM output is", out5.size())
        out6 = self.up6(out5)
        out7 = self.up7(out6)
        out8 = self.up8(out7)
        out9 = self.up9(out8)


        return out9

class Sensor(nn.Module):
    def __init__(self, classes, input_channels):
        super(Sensor, self).__init__()
        self.classes = classes
        self.input_channels = input_channels


        self.unetDown = UNETDown(self.input_channels)
        self.unetUp = UNETUp(self.classes)

        self.tdDown = TimeDistributed(self.unetDown)
        self.blstm1 = ConvBLSTM(in_channels=512, hidden_channels=1024, kernel_size=(3, 3), batch_first=True)
        self.tdUp = TimeDistributed(self.unetUp)
        self.blstm2 = ConvBLSTM(in_channels=64, hidden_channels=64, kernel_size=(3, 3), batch_first=True)
        self.outconv = OutConv(64, 1)



    def forward(self, images):

        out = self.tdDown(images)

        rev_index = list(reversed([i for i in range(out.size(1))]))
        reversed_out = out[:,rev_index,...]

        out = self.blstm1(out, reversed_out)

        out = self.tdUp(out)

        rev_index = list(reversed([i for i in range(out.size(1))]))
        reversed_out = out[:,rev_index,...]

        out = self.blstm2(out, reversed_out)

        out = torch.sum(out, dim=1)

        out = self.outconv(out)


        return out









        

        




