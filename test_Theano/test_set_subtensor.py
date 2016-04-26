import theano
import numpy as np
from theano import tensor as T

size=10
x=T.ivector('x')
y=T.matrix('y')

y=theano.shared(np.random.rand(3,3))
# y=theano.shared(np.random.randint(3,3),dtype=np.int32)

ff=theano.function([x],updates=[(y,T.set_subtensor(y[0,:],x))])
# ff1=theano.function([x],updates=[(y,T.set_subtensor(y[(0,0)],x))])
input_x=np.array([3.0,3,3])
input_x1=np.array([0])
# ff1(input_x1)
ff(input_x)
print y.get_value()