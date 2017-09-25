#coding=utf8
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms

len=20
input_size=30
hidden_size=50
n_classes=3

c1=np.random.randn(len,input_size)-3
c2=np.random.randn(len,input_size)
c3=np.random.randn(len,input_size)+3


class Lstm(nn.Module):
    def __init__(self):
        super(Lstm).__init__()
        self.layer=nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
        )
        self.Linear=nn.Linear(hidden_size,n_classes)

    def forward(self,x):
        x=self.layer(x)


