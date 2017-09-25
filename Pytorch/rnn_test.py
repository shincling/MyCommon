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
batch_size=16
n_sample=300

c1=np.random.randn(n_sample,len,input_size)-3
c2=np.random.randn(n_sample,len,input_size)
c3=np.random.randn(n_sample,len,input_size)+3

y_c1=np.int32(np.zeros(n_sample)-1)
y_c2=np.int32(np.ones(n_sample))
y_c3=np.int32(np.ones(n_sample)+1)

c1=torch.FloatTensor(c1)
c2=torch.FloatTensor(c1)
c3=torch.FloatTensor(c1)
y_c1=torch.IntTensor(y_c1)
y_c2=torch.IntTensor(y_c2)
y_c3=torch.IntTensor(y_c3)
x_batch=torch.cat((c1,c2,c3),0)
y_batch=torch.cat((y_c1,y_c2,y_c3),0)


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
        out=self.Linear(x[:,-1])
        return out

for epoch in range(10):
    for batch in list(x_batch[:batch_size]):
        print 'hh'


