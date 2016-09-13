__author__ = 'shin'
import fpectl
from keras.models import Sequential
from keras.layers import Merge, LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
nb_classes = 10

encoder_a = Sequential()
encoder_a.add(LSTM(32, input_shape=(timesteps, data_dim)))
# encoder_a.add(LSTM(32, input_shape=(timesteps, data_dim),return_sequences=True))

encoder_b = Sequential()
encoder_b.add(LSTM(32, input_shape=(timesteps, data_dim)))
# encoder_b.add(LSTM(32, input_shape=(timesteps, data_dim),return_sequences=True))

decoder = Sequential()
decoder.add(Merge([encoder_a, encoder_b], mode='concat'))
decoder.add(Dense(32, activation='relu'))
decoder.add(Dense(nb_classes, activation='softmax'))

decoder.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

# generate dummy training data
x_train_a = np.random.random((1000, timesteps, data_dim))
x_train_b = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, nb_classes))
x_train_a=np.float32(x_train_a)
x_train_b=np.float32(x_train_b)
y_train=np.float32(y_train)
# generate dummy validation data
x_val_a = np.random.random((100, timesteps, data_dim))
x_val_b = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, nb_classes))
x_val_a=np.float32(x_train_a)
x_val_b=np.float32(x_train_b)
y_val=np.float32(y_train)
# fpectl.turnon_sigfpe()
decoder.fit([x_train_a, x_train_b], y_train,
            batch_size=10000, nb_epoch=15,
            validation_data=([x_val_a, x_val_b], y_val))
# decoder.fit([x_train_a, x_train_b], y_train,
#             batch_size=64, nb_epoch=15)
