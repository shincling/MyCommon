#coding=utf8
from collections import OrderedDict
import numpy as np
import pickle

emb_source='bdbk_standard'
vocab_dict=OrderedDict()
vocab_wordlist=[]
vocab_emblist=[]
# vocab_list=np.loadtxt(emb_source,dtype=str,delimiter=' ',usecols=[0])
# vocab_emb=np.loadtxt(emb_source,delimiter=' ',usecols=range(1,101))
# print len(vocab_list)#,len(vocab_emb)
# print ','.join(vocab_list[:30])
ff=open(emb_source,'r').readlines()
print len(ff)
for i in ff[:]:
    np_emb=np.zeros(100)
    partindex=i.find(' ')
    word=i[:partindex]
    embedding=map(float,i[(partindex+1):-2].split(' '))
    np_emb[:]=embedding
    # vocab_dict[word]=np_emb
    vocab_wordlist.append(word)
    vocab_emblist.append(np_emb)
print len(vocab_wordlist)
print len(vocab_emblist)
# pickle.dump(vocab_dict,open('word2vec_dict_20160603','w'))
pickle.dump(vocab_wordlist,open('word2vec_wordlist_20160603','w'))
print 'finish the word list'
pickle.dump(vocab_emblist,open('word2vec_emblist_20160603','w'))
