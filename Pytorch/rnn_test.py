#coding=utf8
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
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

y_c1=np.int32(np.zeros(n_sample))
y_c2=np.int32(np.ones(n_sample))
y_c3=np.int32(np.ones(n_sample)+1)

c1=torch.FloatTensor(c1)
c2=torch.FloatTensor(c2)
c3=torch.FloatTensor(c3)
y_c1=torch.IntTensor(y_c1)
y_c2=torch.IntTensor(y_c2)
y_c3=torch.IntTensor(y_c3)
x_batch=torch.cat((c1,c2,c3),0)
y_batch=torch.cat((y_c1,y_c2,y_c3),0)

torch_dataset = Data.TensorDataset(data_tensor=x_batch, target_tensor=y_batch)
data=Data.DataLoader(torch_dataset,batch_size,True)
class Lstm(nn.Module):
    def __init__(self):
        super(Lstm,self).__init__()
        self.layer=nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.Linear=nn.Linear(hidden_size,n_classes)

    def forward(self,x):
        x,_=self.layer(x)
        out=self.Linear(x[:,-1,:])
        return out

net=Lstm()
print net
opt=torch.optim.Adam(net.parameters())
los=nn.CrossEntropyLoss()
for epoch in range(10):
    sum=0
    for idx,(xx,yy) in enumerate(data):
        x=Variable(xx)
        y=Variable(yy)
        predict=net(x)
        out=torch.max(predict,1)[1]
        right_n=torch.sum(out==y)
        sum+=np.int32(right_n.data.numpy()) #这块一定注意，right_n是一个byteTensor的东西，最多就是256，超过256直接归零，非常有问题！！！
        loss=los(predict,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc=right_n.data.numpy()[0]/float(xx.shape[0])
        print 'epoch:{},idx:{},loss:{},acc:{}'.format(epoch,idx,loss.data.numpy(),acc)
    print '\n\n epoch acc:',sum/float(n_sample*3),sum



