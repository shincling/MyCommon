import torch
import torch.nn as nn
from torch import autograd
m=nn.Conv2d(3,10,5,stride=2)
input=autograd.Variable(torch.randn(16,3,20,30))
output1=m(input)
print output1.data.shape

m=nn.Conv3d(3,10,5,stride=2)
input=autograd.Variable(torch.randn(16,3,20,30,40))
output2=m(input)
print output2.data.shape
