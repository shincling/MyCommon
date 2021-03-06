#coding=utf8
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)    # reproducible

# Hyper Parameters
TIME_STEP = 10      # rnn time step
INPUT_SIZE = 1      # rnn input size
LR = 0.02           # learning rate

# show data
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps)    # float32 for converting torch FloatTensor
y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label='target (cos)')
plt.plot(steps, x_np, 'b-', label='input (sin)')
plt.legend(loc='best')
plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state) #shin:这个RNN的h_t确实都是只是最后一个step的东西。

        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # return outs, h_state

rnn = RNN().cuda()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()

h_state = None      # for initial hidden state

plt.figure(1, figsize=(12, 5))
plt.ion()           # continuously plot

dynamic=1
d_step=0
for step in range(60):
    if dynamic:
        dynamic_steps = np.random.randint(1, 4)  # 随机 time step 长度
        start, end = d_step * np.pi, (d_step + dynamic_steps) * np.pi  # different time steps length
        d_step += dynamic_steps

        # use sin predicts cos
        steps = np.linspace(start, end, 10 * dynamic_steps, dtype=np.float32)
        print dynamic_steps
    else:
        start, end = step * np.pi, (step+1)*np.pi   # time range
        # use sin predicts cos
        steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)    # float32 for converting torch FloatTensor
    y_np = np.cos(steps)

    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis])).cuda()    # shape (batch, time_step, input_size)
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis])).cuda()

    prediction, h_state = rnn(x, h_state)   # rnn output
    # !! next step is important !!
    if 1:
        h_state = Variable(h_state.data)        # repack the hidden state, break the connection from last iteration
    else:
        h_state = None # 这是一个很好的测试，如果每一个初试h_state都是None,那么每一个sin开始的那个地方，它不知道往上还是往下，必须得有个数据之后才知道。所以学习初来一个很有意思的曲线。

    loss = loss_func(prediction, y)         # cross entropy loss
    optimizer.zero_grad()                   # clear gradients for this training step
    optimizer.param_groups[0]['lr']=0.01 # shin: 可以用这种方式来设定学习率，在train的过程中变化。
    loss.backward()                         # backpropagation, compute gradients
    optimizer.step()                        # apply gradients

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.cpu().numpy().flatten(), 'b-')
    plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()