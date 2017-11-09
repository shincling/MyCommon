import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import myNet

torch.manual_seed(1)

# class VIDEO_QUERY(nn.Module):
#     def __init__(self,total_frames,video_size):
#         self.images_net=models.inception_v3(pretrained=True)
#
#     def forward(self,x):

# mm=models.inception_v3()
mm=myNet.inception_v3(1)
#
xx=Variable(torch.rand([2,3,300,300]))
print mm(xx)[2].size()



