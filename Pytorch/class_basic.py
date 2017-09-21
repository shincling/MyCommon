#coding=utf8
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(11)    # reproducible

# make fake data
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), 0).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden1 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        if 1:# share weights of the two layer
            self.hidden2.weight=self.hidden1.weight
            self.hidden2.bias=self.hidden1.bias
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        # x = F.relu(self.hidden1(F.relu(self.hidden1(F.relu(self.hidden(x))))))      # activation function for hidden layer
        # x = F.relu(self.hidden1(F.relu(self.hidden(x))))      # activation function for hidden layer
        x = F.relu(self.hidden2(F.relu(self.hidden1(F.relu(self.hidden(x))))))      # activation function for hidden layer
        # x = F.relu(self.hidden1(F.relu(self.hidden1(F.relu(self.hidden1(x))))))      # activation function for hidden layer
        x = self.out(x)
        return x

net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2)
)

if 1:
    net = Net(n_feature=2, n_hidden=10, n_output=2)     # define the network
else:
    net=net2

print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
cc=net.parameters()
for i in cc:
    print i
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
# loss_func = torch.nn.CrossEntropyLoss(torch.FloatTensor([99,1])) #这个数字是带权重的更新，对于数据有不平衡的时候有很好的效果 # the target label is NOT an one-hotted

plt.ion()   # something about plotting

tt=time.time()
for t in range(100):
    out = net(x)                 # input x and predict based on x
    # print out
    loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200.
        print accuracy
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

for i in net.parameters():
    print i
print 'cost time:',time.time()-tt
plt.ioff()
plt.show()