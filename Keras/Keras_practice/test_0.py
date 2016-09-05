__author__ = 'shin'
import numpy
from keras.models import Sequential
from keras.layers import Dense, Activation

def make_data():
    data=numpy.zeros(10,10000)
    for i in range(10):
       for j in range(1000):
           data[i][j]=numpy.array([0]+[numpy.random.random]*9)
    return data


model = Sequential([
Dense(32, input_dim=10),
Activation('relu'),
Dense(10),
Activation('softmax'),
])

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])



