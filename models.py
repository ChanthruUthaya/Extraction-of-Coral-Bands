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

        c10_out = torch.sigmoid(self.l10conv1(c94_out))
        
        return c10_out

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.constant_(layer.bias, 0.1)
        if hasattr(layer, "weight"):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)