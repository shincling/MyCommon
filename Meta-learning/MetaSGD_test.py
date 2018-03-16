#coding=utf8
import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy

seed = 0
plot = True
innerstepsize = 0.02 # stepsize in inner SGD
innerepochs = 1 # 不是k number of epochs of each inner SGD
outerstepsize0 = 0.2 # stepsize of outer optimization, i.e., meta-optimization
outerstepsize = 0.02 # stepsize of outer optimization, i.e., meta-optimization
niterations = 30000 # number of outer updates; each iteration we sample one task and update on it
test_num=20 #一个task里面 test的占总共（50个点）的比例

rng = np.random.RandomState(seed)
torch.manual_seed(seed)

# Define task distribution
x_all = np.linspace(-5, 5, 50)[:,None] # All of the x points
ntrain = 10 # Size of training minibatches 这个就是batch版本程序的n没跑了
# ntrain = 20 # Size of training minibatches
def gen_task():
    "Generate classification problem"
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x : np.sin(x + phase) * ampl
    return f_randomsine

# Define model. Reptile paper uses ReLU, but Tanh gives slightly better results
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1),
) # 整个model模拟的是asin(x+b)函数的

class Adaptation(torch.nn.Module):
    def __init__(self):
        super(Adaptation,self).__init__()
        self.alpha_1=nn.Parameter(torch.rand(64,1))
        self.alpha_1b=nn.Parameter(torch.rand(64))
        self.alpha_2=nn.Parameter(torch.rand(64,64))
        self.alpha_2b=nn.Parameter(torch.rand(64))
        self.alpha_3=nn.Parameter(torch.rand(1,64))
        self.alpha_3b=nn.Parameter(torch.rand(1))
        
        # self.alpha_1=torch.autograd.Variable(torch.rand(64,1),requires_grad=True)
        # self.alpha_1b=torch.autograd.Variable(torch.rand(64),requires_grad=True)
        # self.alpha_2=torch.autograd.Variable(torch.rand(64,64),requires_grad=True)
        # self.alpha_2b=torch.autograd.Variable(torch.rand(64),requires_grad=True)
        # self.alpha_3=torch.autograd.Variable(torch.rand(1,64),requires_grad=True)
        # self.alpha_3b=torch.autograd.Variable(torch.rand(1),requires_grad=True)

    def forward(self, model):
        for param,alpha in zip(model.parameters(),[self.alpha_1,self.alpha_1b,self.alpha_2,self.alpha_2b,self.alpha_3,self.alpha_3b,]):
            # print 'prarm shape:',param.data.shape,'alpha shape:',alpha.data.shape
            param.data=param.data-alpha.data*param.grad.data
            # param=param-alpha
        # return model

ada=Adaptation()

def totorch(x):
    return ag.Variable(torch.Tensor(x))

def train_on_batch(x, y):
    x = totorch(x)
    y = totorch(y)
    model.zero_grad()
    ypred = model(x)
    loss = (ypred - y).pow(2).mean()
    loss.backward()
    # for param in model.parameters():
    #     print '111'
    #     print param[:3]
    #     break
    ada(model)
    # for param in model.parameters():
    #     print '222'
    #     print param[:3]
    #     break
    for param in ada.parameters():
        print 'alpha alpha alpha'
        print param[:3]

    train_param_grad_list=[]
    for param in model.parameters():
        train_param_grad_list.append(param.grad.data)
    return train_param_grad_list

def predict(x):
    x = totorch(x)
    return model(x).data.numpy()

# Choose a fixed task and minibatch for visualization
f_plot = gen_task()
xtrain_plot = x_all[rng.choice(len(x_all), size=ntrain)]

# Reptile training loop
for iteration in range(niterations):
    weights_before = deepcopy(model.state_dict())
    # Generate task
    f = gen_task()
    y_all = f(x_all)
    # Do SGD on this task
    inds = rng.permutation(len(x_all))
    train_inds=inds[:-test_num]
    test_inds=inds[-test_num:]

    for _ in range(innerepochs):
        train_grads_list = train_on_batch(x_all[train_inds], y_all[train_inds])

        model.zero_grad()
        ypred = model(totorch(x_all[test_inds]))
        loss = (ypred - totorch(y_all[test_inds])).pow(2).mean()
        loss.backward()
        # print 'hhh'
        # print ada.alpha_1.grad
        # for param in ada.parameters():
        #     print 'hhh'
        #     print param.grad
        #     1/0
        for param,alpha,train_grad in zip(model.parameters(),ada.parameters(),train_grads_list):
            param.data -= innerstepsize * param.grad.data
            alpha.data += innerstepsize * train_grad * param.grad.data

    # weights_after = model.state_dict()
    # # outerstepsize = outerstepsize0 * (1 - iteration / niterations) # linear schedule
    # model.load_state_dict({name :
    #     weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize
    #     for name in weights_before})

    # Periodically plot the results on a particular task and minibatch
    if plot and iteration==0 or (iteration+1) % 1000 == 0:
        plt.cla()
        f = f_plot
        weights_before = deepcopy(model.state_dict()) # save snapshot before evaluation
        plt.plot(x_all, predict(x_all), label="pred after 0", color=(0,0,1))
        for inneriter in range(32):
            train_on_batch(xtrain_plot, f(xtrain_plot))
            if (inneriter+1) % 8 == 0:
                frac = (inneriter+1) / 32
                plt.plot(x_all, predict(x_all), label="pred after %i"%(inneriter+1), color=(frac, 0, 1-frac))
        plt.plot(x_all, f(x_all), label="true", color=(0,1,0))
        lossval = np.square(predict(x_all) - f(x_all)).mean()
        plt.plot(xtrain_plot, f(xtrain_plot), "x", label="train", color="k")
        plt.ylim(-4,4)
        plt.legend(loc="lower right")
        plt.pause(0.01)
        model.load_state_dict(weights_before) # restore from snapshot
        print "-----------------------------"
        print "iteration               {}".format(iteration+1)
        print "loss on plotted curve   {}".format(lossval) # would be better to average loss over a set of examples, but this is optimized for brevity