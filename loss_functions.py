import torch
import torch.backends.cudnn
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from multiprocessing import cpu_count
from pathlib import Path
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class WeightedCrossEntropyLoss(nn.BCEWithLogitsLoss):
    def __init__(self, reduction: str):
        super(WeightedCrossEntropyLoss, self).__init__(reduction=reduction)
    
    def forward(self, inputs: torch.Tensor, target: torch.Tensor, weights:torch.Tensor) -> torch.Tensor:

        assert self.weight is None or isinstance(self.weight, Tensor)
        assert self.pos_weight is None or isinstance(self.pos_weight, Tensor)

        # loss = torch.Tensor([])

        # for i, _ in enumerate(inputs):

        loss_val = F.binary_cross_entropy_with_logits(inputs, target,
                                                self.weight,
                                                pos_weight=weights,
                                                reduction=self.reduction)
            # print(loss_val)
            # loss = torch.cat([loss, loss_val])
        
        # loss = torch.mean(loss)
        # loss = torch.div(loss, 256*256)

        print(loss_val)

        return loss_val
    

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        #self.alpha = 0 if alpha is None else alpha
        self.eps = 1e-8
        self.reduction = reduction
        self.alpha =0

    
    def forward(self, input, target):

        probs = torch.sigmoid(input)

        #target = target.unsqueeze(dim=1)

        loss_tmp = -torch.pow((1. - probs), self.gamma) * target * torch.log(probs + self.eps) \
                -torch.pow(probs, self.gamma) * (1. - target) * torch.log(1. - probs + self.eps)

        loss_tmp = loss_tmp.squeeze(dim=1)

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        
        return loss

# self.alpha (1 - self.alpha) * 