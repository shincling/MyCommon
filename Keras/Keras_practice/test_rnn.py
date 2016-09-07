__author__ = 'shin'
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,Embedding,Dropout,LSTM

maxlen=111
model=Sequential()
model.add(Embedding(input_dim=30,output_dim=50,input_length=maxlen))
model.add(LSTM(output_dim=50,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=maxlen))

