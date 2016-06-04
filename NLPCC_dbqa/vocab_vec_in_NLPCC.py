#coding=utf8
from collections import OrderedDict
import numpy as np
import pickle

vocab_NLPCC=pickle.load(open('vocabSet_in_NLPCC'))
vocab_total=pickle.load(open('word2vec_wordlist_20160603'))
print 'vocab_nlpcc:{}'.format(len(vocab_NLPCC))
print 'vocab_total:{}'.format(len(vocab_total))

cover_idx=0
for word in vocab_NLPCC:
    if word.encode('utf8') in vocab_total:
       cover_idx+=1
print float(cover_idx)/len(vocab_NLPCC)

'''
vocab_nlpcc:232476
vocab_total:1142896
0.479464546878
'''