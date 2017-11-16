#coding=utf8
import torch
from torch.autograd import Variable
import numpy as np

def mul3(x):
    return x*3
def add3(x):
    # return x+3
    return x+Variable(torch.ones([2,2]))
    # return x+torch.FloatTensor(np.array([[1,2],[10,11]]))
def mul3_numpy(x):
    return Variable(torch.FloatTensor(x.data.numpy()*3))
a=Variable(torch.ones(2,2),requires_grad=True)
b=a*2
c=b*b
# c=mul3(c)#即便经过了一个函数，也是能够反传的
c=add3(c)#即便经过了一个函数，也是能够反传的,但是这个常数必须用Variable
c[0]=c[0]+Variable(torch.ones(2))
c[0]=c[0]*2
# c=mul3_numpy(c)#经过了一个函数但是改变了原来的性质(从Variable变到了numpy，这样就出问题了)
z=c.mean()
r=c.sum()

print z

r.backward()
# r.backward(torch.Tensor([[1,2],[3,4]]))
print a.grad
print b.grad
print c.grad

# r.backward(torch.Tensor([[1,2],[3,4]]))
# print a.grad

