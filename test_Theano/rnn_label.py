import theano
import theano.tensor as T
import lasagne
import numpy as np

def create_sequence(total,len,embedding_size):
    data=np.random.randint(0,2,[total,len,embedding_size])
    return data

hidden_size=50
hidden_size=1
output_size=1
batch_size=32
total_epoch=100
length=10
num_train=10000
num_test=1000

X=T.tensor3()
Y=T.imatrix()
# Y=T.iscalar()
l_in=lasagne.layers.InputLayer(shape=(None,None,hidden_size))
rnn=lasagne.layers.GRULayer(l_in,output_size,only_return_final=True)
output=lasagne.layers.flatten(rnn)
# output=lasagne.layers.DenseLayer(rnn,1,nonlinearity=lasagne.nonlinearities.sigmoid)

result=lasagne.layers.helper.get_output(output,{l_in:X})
cost=T.nnet.binary_crossentropy(result,Y).sum()
params = lasagne.layers.helper.get_all_params(output, trainable=True)
grads = T.grad(cost, params)
updates = lasagne.updates.rmsprop(grads, params, learning_rate=0.0005)

train_model = theano.function([X,Y], [cost,result], updates=updates)


num_batch_train=num_train/batch_size
num_batch_test=num_test/batch_size
train_data=create_sequence(num_train,length,1)
train_data[:int(0.1*num_train)]=np.zeros([length,1])
for _ in range(total_epoch):
    np.random.shuffle(train_data)
    for idx_batch in range(num_batch_train):
        x=train_data[idx_batch*batch_size:(idx_batch+1)*batch_size]
        y_label=np.int32(np.count_nonzero(x,axis=1)==0)
        cost,result=train_model(x,y_label)
        print 'cost:{},result:{}'.format(cost,result.flatten())
        print 'y_label:',y_label.flatten()





