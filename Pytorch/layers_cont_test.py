import torch
import torch.nn as nn
from torch.autograd import variable as V
layer1=nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2)
)
layer2=nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2)
)
layer3=nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2)
)

class N1(nn.Module):
    def __init__(self):
        super(N1,self).__init__()
        self.l1=torch.nn.Linear(2, 20)
        self.l2=torch.nn.ReLU()
        self.l3=torch.nn.Linear(20, 2)
    def forward(self,x):
        return self.l3(self.l2(self.l1(x)))

print N1
n1=N1()
net_all=nn.Sequential(
    layer1,
    layer2,
    layer3,
    n1
)

print torch.nn.Linear(2,3)
print net_all
print net_all.state_dict().keys()
