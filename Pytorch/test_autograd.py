import torch
from torch.autograd import Variable

a=Variable(torch.ones(2,2),requires_grad=True)
b=a*2
c=b*b
z=c.mean()
r=c.sum()

print z

z.backward()
# r.backward(torch.Tensor([[1,2],[3,4]]))
print a.grad
print b.grad
print c.grad

# r.backward(torch.Tensor([[1,2],[3,4]]))
# print a.grad

