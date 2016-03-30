from __future__ import division
from keras import backend as K
from keras.layers.core import MaskedLayer
try:
    import cPickle as pickle
except ImportError:
    import pickle


class Bidirectional(MaskedLayer):
    ''' Bidirectional wrapper for RNNs

    # Arguments:
        rnn: `Recurrent` object.
        merge_mode: Mode by which outputs of the forward and reverse RNNs will be combined. One of {sum, mul, concat, ave}

    # Examples:
    ```python
    model = Sequential()
    model.add(Bidirectional(LSTM(10, input_shape=(10, 20))))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit([X_train,], Y_train, batch_size=32, nb_epoch=20,
              validation_data=([X_test], Y_test))
    ```
    '''
    def __init__(self, rnn, merge_mode='concat', weights=None):

        self.forward = rnn
        self.reverse = pickle.loads(pickle.dumps(rnn))
        self.merge_mode = merge_mode
        if weights:
            nw = len(weights)
            self.forward.initial_weights = weights[:nw//2]
            self.reverse.initial_weights = weights[nw//2:]
        self._cache_enabled = True
        self.stateful = rnn.stateful
        self.return_sequences = rnn.return_sequences
        if hasattr(rnn, '_input_shape'):
            self._input_shape = rnn.input_shape
        elif hasattr(rnn, 'previous') and rnn.previous:
            self.previous = rnn.previous

    def get_weights(self):
        return self.forward.get_weights() + self.reverse.get_weights()

    def set_weights(self, weights):
        nw = len(weights)
        self.forward.set_weights(weights[:nw//2])
        self.reverse.set_weights(weights[nw//2:])

    def set_previous(self, layer):
        self.previous = layer
        self.forward.set_previous(layer)
        self.reverse.set_previous(layer)
        self._input_shape = layer.output_shape

    @property
    def cache_enabled(self):
        return self._cache_enabled

    @cache_enabled.setter
    def cache_enabled(self, value):
        self._cache_enabled = value
        self.forward.cache_enabled = value
        self.reverse.cache_enabled = value

    @property
    def output_shape(self):
        if self.merge_mode in ['sum', 'ave', 'mul']:
            return self.forward.output_shape
        elif self.merge_mode == 'concat':
            shape = list(self.forward.output_shape)
            shape[-1] *= 2
            return tuple(shape)

    def get_output(self, train=False):
        X = self.get_input(train) # 0,0,0,1,2,3,4
        mask = self.get_input_mask(train) # 0,0,0,1,1,1,1

        def reverse(x):
            if K.ndim == 2:
                x = K.expand_dims(x, -1)
                rev = K.permute_dimensions(x, (1, 0, 2))[::-1]
                rev = K.squeeze(rev, -1)
            else:
                rev = K.permute_dimensions(x, (1, 0, 2))[::-1]                
            return K.permute_dimensions(rev, (1, 0, 2))

        X_rev = reverse(X) # 4,3,2,1,0,0,0
        Y = self.forward(X, mask) # 0,0,0,1,3,6,10
        mask_rev = reverse(mask) if mask else None # 1,1,1,1,0,0,0
        Y_rev = self.reverse(X_rev, mask_rev) # 4,7,9,10,10,10,10

        #Fix allignment
        if self.return_sequences:
            Y_rev = reverse(Y_rev) # 10,10,10,10,9,7,4

        if self.merge_mode == 'concat':
            return K.concatenate([Y, Y_rev])
        elif self.merge_mode == 'sum':
            return Y + Y_rev
        elif self.merge_mode == 'ave':
            return (Y + Y_rev) / 2
        elif self.merge_mode == 'mul':
            return Y * Y_rev

    def get_output_mask(self, train=False):
        if self.forward.return_sequences:
            return self.get_input_mask(train)
        else:
            return None

    @property
    def input_shape(self):
        return self.forward.input_shape

    def get_input(self, train=False):
        return self.forward.get_input(train)

    @property
    def non_trainable_weights(self):
        return self.forward.non_trainable_weights + self.reverse.non_trainable_weights

    @property
    def trainable_weights(self):
        return self.forward.trainable_weights + self.reverse.trainable_weights

    @trainable_weights.setter
    def trainable_weights(self, weights):
        nw = len(weights)
        self.forward.trainable_weights = weights[:nw//2]
        self.reverse.trainable_weights = weights[nw//2:]

    @non_trainable_weights.setter
    def non_trainable_weights(self, weights):
        nw = len(weights)
        self.forward.non_trainable_weights = weights[:nw//2]
        self.reverse.non_trainable_weights = weights[nw//2:]


    @property
    def regularizers(self):
        return self.forward.get_params()[1] + self.reverse.get_params()[1]

    @property
    def constraints(self):
        return self.forward.get_params()[2] + self.reverse.get_params()[2]

    @property
    def updates(self):
        return self.forward.get_params()[3] + self.reverse.get_params()[3]

    def reset_states(self):
        self.forward.reset_states()
        self.reverse.reset_states()

    def build(self):
        if not hasattr(self.forward, '_input_shape'):
            if hasattr(self, '_input_shape'):
                self.forward._input_shape = self._input_shape
                self.reverse._input_shape = self._input_shape
                self.forward.previous = self.previous
                self.reverse.previous = self.previous
                self.forward.trainable_weights = []
                self.reverse.trainable_weights = []
                self.forward.build()
                self.reverse.build()

    def get_config(self):
        config = {
                  "name": self.__class__.__name__,
                  "rnn": self.forward.get_config(),
                  "merge_mode": self.merge_mode}
        base_config = super(Bidirectional, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
