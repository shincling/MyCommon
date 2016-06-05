#coding=utf8
from collections import OrderedDict
import numpy as np
import pickle

vocab_NLPCC=pickle.load(open('vocabSet_in_NLPCC'))
# vocab_total=pickle.load(open('word2vec_wordlist_20160603'))
vocab_total=pickle.load(open('word2vec_dict_20160603'))
print 'vocab_nlpcc:{}'.format(len(vocab_NLPCC))
print 'vocab_total:{}'.format(len(vocab_total))
nlpcc_dict={}
cover_idx=0
for word in vocab_NLPCC:
    if word.encode('utf8') in vocab_total:
        cover_idx+=1
        nlpcc_dict[word.encode('utf8')]=vocab_total[word.encode('utf8')]
    else:
        nlpcc_dict[word.encode('utf8')]=np.random.normal(0,1,[3,])
print 'cover_idx:',len(cover_idx),float(cover_idx)/len(vocab_NLPCC)

print 'nlpcc_dict:',len(nlpcc_dict)

pickle.dump(nlpcc_dict,open('nlpcc_dict_20160605','w'))
# print nlpcc_dict
'''
vocab_nlpcc:232476
vocab_total:1142896
0.479464546878
'''