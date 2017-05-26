__author__ = 'shin'
import numpy
from keras.models import Sequential
from keras.layers import Dense, Activation, Input
from keras.models import Model

def make_data():
    data=numpy.zeros([10000,10])
    for i in range(10):
       for j in range(1000):
           data[i*1000+j]=numpy.array([i]+[numpy.random.random()]*9)
    return data

print '2hhhhh'
print 'hhhh'

# model = Sequential([
# Dense(32, input_dim=10),
# Activation('relu'),
# Dense(10),
# Activation('softmax'),
# ])

input=Input(shape=[10])
layer=Dense(32)(input)
layer=Activation('relu')(layer)
layer=Dense(10)(layer)
layer=Activation('softmax')(layer)
model=Model(inputs=input,outputs=layer)

cc=make_data()
labels=numpy.zeros_like(cc)
for idx,line in enumerate(labels):
    line[int(cc[idx,0])]=1
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(cc,labels,batch_size=10,nb_epoch=30)

# bb=model.get_weights()
# pass


