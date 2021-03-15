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

class LSTMLayer(nn.Module):
    def __init__(self, input_dim = 16, hidden_dim= 16, layers=1):
        super(LSTMLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstmlayer = nn.LSTM(input_dim, hidden_dim, batch_first = True)
        self.output_buffer = []

    
    def forward(self, images: torch.Tensor):
        
        hidden_state = torch.zeros(1, 1024,16).to(DEVICE) #(num_layers*num_directions, batch, input_size)
        cell_state = torch.zeros(1, 1024,16).to(DEVICE)


        self.init_buffer()
        for image in images:
            out, (hidden_state, cell_state) = self.lstmlayer(image, (hidden_state, cell_state))
            out = out.squeeze(0).detach()
            self.output_buffer.append(out)
            del out
            
        
        return torch.stack(self.output_buffer, dim=0)
    
    def init_buffer(self):
        if len(self.output_buffer) > 0:
            self.output_buffer = []



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
    
    
class UNET2DLSTM(nn.Module):
    def __init__(self, classes:int, height:int, width:int, channels:int):
        super().__init__()

        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.classes = classes

        self.down1 = DoubConv(self.input_shape.channels, 64)
        self.down2 = DownLayer(64, 128)
        self.down3 = DownLayer(128, 256)
        self.down4 = DownLayer(256,512)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.down5 = DownLayer(512,1024)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.lstm = LSTMLayer()
        self.up6 = UpLayer(1024, 512, (0,1,0,1))
        self.up7 = UpLayer(512, 256, (0,1,0,1))
        self.up8 = UpLayer(256, 128, (0,1,0,1))
        self.up9 = nn.Sequential(
                    UpLayer(128, 64, (0,1,0,1)),
                    nn.Conv2d(in_channels = 64,
                     out_channels = self.classes, 
                     kernel_size = (3,3),
                     padding = (1,1)),
                     nn.BatchNorm2d(self.classes),
                    nn.ReLU(inplace=True)
        )
        self.out10 = OutConv(self.classes, 1)

    def forward(self, images: torch.Tensor):
        out1 = self.down1(images)
        out2 = self.down2(out1)
        out3 = self.down3(out2)

        out4 = self.down4(out3)
        out4 = self.dropout1(out4)
    

        out5 = self.down5(out4)
        out5 = self.dropout2(out5)

        features = out5.view(-1,1024,16, 16)
        #print("input to lstm", features.size()) #2,1024,16, 16
        features = self.lstm(features)
        #outlstm = features.view(-1,1024,16,16)

       # print("lstm out ", features.size())

        out5 = features.view(-1,1024,16,16)

        #print("lstm out 2 ", out5.size())


        out6 = self.up6(out5)
        out7 = self.up7(out6)
        out8 = self.up8(out7)
        out9 = self.up9(out8)

        out = self.out10(out9)

        return out

class UNET2D(nn.Module):
    def __init__(self, classes:int, height:int, width:int, channels:int):
        super().__init__()

        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.classes = classes

        self.down1 = DoubConv(self.input_shape.channels, 64)
        self.down2 = DownLayer(64, 128)
        self.down3 = DownLayer(128, 256)
        self.down4 = DownLayer(256,512)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.down5 = DownLayer(512,1024)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.lstm = LSTMLayer()
        self.up6 = UpLayer(1024, 512, (0,1,0,1))
        self.up7 = UpLayer(512, 256, (0,1,0,1))
        self.up8 = UpLayer(256, 128, (0,1,0,1))
        self.up9 = nn.Sequential(
                    UpLayer(128, 64, (0,1,0,1)),
                    nn.Conv2d(in_channels = 64,
                     out_channels = self.classes, 
                     kernel_size = (3,3),
                     padding = (1,1)),
                     nn.BatchNorm2d(self.classes),
                    nn.ReLU(inplace=True)
        )
        self.out10 = OutConv(self.classes, 1)

    def forward(self, images: torch.Tensor):
        out1 = self.down1(images)
        out2 = self.down2(out1)
        out3 = self.down3(out2)

        out4 = self.down4(out3)
        out4 = self.dropout1(out4)
    

        out5 = self.down5(out4)
        out5 = self.dropout2(out5)

        out6 = self.up6(out5)
        out7 = self.up7(out6)
        out8 = self.up8(out7)
        out9 = self.up9(out8)

        out = self.out10(out9)

        return out