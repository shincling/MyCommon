import theano
import numpy as np
from theano import tensor as T

size=10
x=T.vector('x')
y=T.matrix('y')
x1=T.vector('x1')
x2=T.matrix('all')


y=theano.shared(np.random.rand(3,3))
# y=theano.shared(np.random.randint(3,3),dtype=np.int32)

ff=theano.function([x],updates=[(y,T.set_subtensor(y[0,:],x))],allow_input_downcast=True)
ff1=theano.function([x1],updates=[(y,T.set_subtensor(y[[1],[1,2]],x1))],allow_input_downcast=True)
ff2=theano.function([x2],updates=[(y,T.set_subtensor(y[[0,1,2],:],x2))],allow_input_downcast=True)
input_x=np.array([3.0,3,3])
input_x1=np.array([0.,1])
input_all=np.random.rand(3,3)
ff1(input_x1)
print y.get_value()
ff(input_x)
print y.get_value()
ff2(input_all)
print y.get_value()