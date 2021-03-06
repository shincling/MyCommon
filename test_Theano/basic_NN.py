#coding=utf8
import lasagne
import numpy as np
import theano
import theano.tensor as T
from sklearn.preprocessing import LabelBinarizer,label_binarize

def get_dataset(dimention):
    x=np.random.random((10000,dimention))
    y=np.zeros((10000))
    for idx,i in enumerate(x):
        if 0.3<i[0]<0.7:
            y[idx]=1
        elif i[0]>=0.7:
            y[idx]=2
    return x,y


dimention=10
xx,yy=get_dataset(dimention)
yy=np.int32(yy)
x=T.matrix('x')
y=T.imatrix('y')
x1=T.vector('x1')
x2=T.matrix('all')

n_classes=3
batch_size=16
n_epoch=530


x_shared=theano.shared(np.zeros((batch_size,dimention),dtype=theano.config.floatX),borrow=True)
y_shared=theano.shared(np.zeros((batch_size,1),dtype=np.int32),borrow=True)

l_in = lasagne.layers.InputLayer(shape=(None, 1,dimention))
l_in1=lasagne.layers.DenseLayer(l_in,30,W=lasagne.init.Normal(std=1),nonlinearity=lasagne.nonlinearities.softmax)
l_theta = lasagne.layers.DenseLayer(l_in,3,W=lasagne.init.Normal(std=1))
l_theta = lasagne.layers.BatchNormLayer(l_theta)
print 'batchnorm'
l_mu=lasagne.layers.NonlinearityLayer(l_theta,nonlinearity=lasagne.nonlinearities.softmax)

'''正确的输入写法应该是l_in:x,因为x才是拟定的变量x_shared只是一个输入,但是这里是通用的其实
　　当用x_shared的时候,底下的givens的　x:x_shared其实就没用了'''
probas = lasagne.layers.helper.get_output(l_mu, {l_in: x})
probas = lasagne.layers.helper.get_output(l_mu, {l_in: x_shared})
pred = T.argmax(probas, axis=1)
cost = T.nnet.categorical_crossentropy(probas, y_shared).sum()
params_all = lasagne.layers.helper.get_all_params(l_mu, trainable=True)
'''这里就有意思了，可以通过tag选择训练的参数'''
params = lasagne.layers.helper.get_all_params(l_mu, regularizable=False)

'''这里是另外一种方式，直接从params中移除某个参数，有效果'''
if 0:
    params.pop()
    print params

params = lasagne.layers.helper.get_all_params(l_mu, shin=True)
grads = T.grad(cost, params)
updates = lasagne.updates.sgd(grads, params, learning_rate=0.05)


givens = {
    # x: x_shared,
    # y: y_shared,
    }

train_model = theano.function([], [cost,pred], givens=givens,updates=updates,on_unused_input='ignore',allow_input_downcast=True)

for epoch in range(n_epoch):
    batch_total_number=len(xx)/batch_size
    cost,error_cout=0,0
    for idx_batch in range(batch_total_number):
        x_batch=xx[idx_batch*batch_size:(idx_batch+1)*batch_size]
        target=yy[idx_batch*batch_size:(idx_batch+1)*batch_size]
        y_batch=label_binarize(target,range(n_classes))
        x_shared.set_value(x_batch)
        y_shared.set_value(np.int32(y_batch))
        cost,pred=train_model()
        params_all = lasagne.layers.helper.get_all_param_values(l_mu, trainable=True)
        for pp in params_all:
            print pp
        cost+=cost

        count=np.count_nonzero(pred-target)
        error_cout+=count
    print cost/batch_size
    print 'accuracy:',1-float(error_cout)/(batch_total_number*batch_size)

lasagne.layers.MergeLayer