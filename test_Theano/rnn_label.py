import theano
import theano.tensor as T
import lasagne
import numpy as np
import time
import pickle

def create_sequence(total,len,embedding_size):
    # data=np.random.randint(-5,2,[total,len,embedding_size])
    data=-5+10*(np.random.rand(total,len,embedding_size))
    return data

hidden_size=50
hidden_size=1
output_size=5
batch_size=32
total_epoch=100
length=10
num_train=50000
num_test=1000

X=T.tensor3()
Y=T.imatrix()
# Y=T.iscalar()
l_in=lasagne.layers.InputLayer(shape=(None,None,hidden_size))
rnn=lasagne.layers.GRULayer(l_in,output_size,only_return_final=True)
# output=lasagne.layers.flatten(rnn)
# output=rnn
output=lasagne.layers.DenseLayer(rnn,1,nonlinearity=lasagne.nonlinearities.sigmoid)

result=lasagne.layers.helper.get_output(output,{l_in:X})
cost=T.nnet.binary_crossentropy(result,Y).sum()
params = lasagne.layers.helper.get_all_params(output, trainable=True)
print params
grads = T.grad(cost, params)
updates = lasagne.updates.rmsprop(grads, params, learning_rate=0.001)
# updates = lasagne.updates.sgd(grads, params, learning_rate=0.01)

train_model = theano.function([X,Y], [cost,result], updates=updates)
test_model = theano.function([X], [result])


num_batch_train=num_train/batch_size
num_batch_test=num_test/batch_size
train_data=create_sequence(num_train,length,1)
train_data[:int(0.3*num_train)]=np.zeros([length,1])
for _ in range(total_epoch):
    np.random.shuffle(train_data)
    for idx_batch in range(num_batch_train):
        x=train_data[idx_batch*batch_size:(idx_batch+1)*batch_size]
        y_label=np.int32(np.count_nonzero(x,axis=1)==0)
        cost,result=train_model(x,y_label)
        print 'cost:{}\n,result:{}'.format(cost,result.flatten()[:10])
        print 'y_label:',y_label.flatten()[:10],'\n\n'

        if str(cost)=='nan' or cost<0.01:
            break
    if str(cost)=='nan' or cost<0.01:
        break

prev_weights = lasagne.layers.helper.get_all_param_values(output,trainable=True)
pickle.dump(prev_weights,open('/home/sw/Shin/Codes/Deep-Rein4cement/One-shot-PGD/Omniglot/params_contRNN_{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),'wb'))
print 'save params !'

test_data=create_sequence(batch_size,length+2,1)
test_data[:5]=np.zeros([length+2,1])
xx=test_data
result=test_model(xx)[0]
print 'test result:{}'.format(result.flatten()[:10])
print 'test_data:',test_data[:10].reshape(10,-1)



