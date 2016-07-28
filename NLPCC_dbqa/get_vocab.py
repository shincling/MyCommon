#coding=utf8
import sys
import re
import pickle
import jieba

input_path='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/nlpcc-iccpol-2016.dbqa.training-data'
# input_path='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/NLPCC2016QA-Update/evatestdata2-dbqa.testing-data-answers'
f_input=open(input_path,'r').readlines()
print 'total lines of input is {}'.format(len(f_input))
sent_set=set()
question_set=set()
ans_set=set()
vocab=set()
for line in f_input:
    each=line.split('\t')
    # question_set.add(each[0])
    # ans_set.add(each[1])
    # if each[1] in ans_set:
    #     print each[1]
    sent_set.add(each[0])
    sent_set.add(each[1])
print 'total num of sents:{}'.format(len(sent_set))
# print len(question_set)
# print len(ans_set)
for sent in sent_set:
    words=jieba._lcut(sent)
    for word in words:
        vocab.add(word)
print len(vocab)
# pickle.dump(vocab,open('vocabSet_in_NLPCC_0701','w'))

