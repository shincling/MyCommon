#coding=utf8
import torch
from torch.autograd import Variable


# 设计思路：有一个需要grad的变量c，把它的部分内容传给一个一开始没有grad的变量rep，
# 然后这个rep去做后面的操作了，得到了梯度，看看是否最终c.grad能否得到梯度，事实证明还是可以的！

cc=Variable(torch.ones(2,2),requires_grad=True)
# dd=(cc*cc).mean()

# dd.backward()
print cc.grad

repalce_cc=Variable(torch.ones(2,2),requires_grad=0) #注意，这里一定得提供一个没有梯度的变量才可以
print repalce_cc.grad

# 到现在位置replace_cc一无所有
repalce_cc[0,:]=cc[0,:]
# repalce_cc[1,:]=cc[0,:]#再多一次传递也没毛病～非常智能
dd=repalce_cc.mean()
dd.backward()
print repalce_cc.grad
print cc.grad



