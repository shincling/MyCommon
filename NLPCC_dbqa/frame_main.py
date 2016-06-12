#coding=utf8
import sys
import re
import pickle
import jieba
def get_vocab(train_file,test_file):
    f_input_train=open(train_file,'r').readlines()
    f_input_test=open(test_file,'r').readlines()
    f_input=f_input_train+f_input_test
    print 'total lines of input is {}'.format(len(f_input))
    sent_set=set()
    vocab=set()
    for line in f_input:
        each=line.split('\t')
        sent_set.add(each[0])
        sent_set.add(each[1])
    print 'total num of sents:{}'.format(len(sent_set))
    for sent in sent_set:
        words=jieba._lcut(sent)
        for word in words:
            vocab.add(word)
    print len(vocab)
    output_name='vocabSet_in_NLPCC_main'
    pickle.dump(vocab,open(output_name,'w'))
    print 'Save the set to {}'.format(output_name)
    return vocab
if __name__=='__main__':
    train_file='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/data_valid/train7_1'
    test_file='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/data_valid/valid3_1'
    build_vocab=True
    if build_vocab:
        get_vocab(train_file,test_file)
