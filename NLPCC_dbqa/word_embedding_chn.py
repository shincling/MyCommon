#coding=utf8
from collections import OrderedDict
import numpy as np

emb_source='bdbk_standard'
vocab_dict=OrderedDict()
# vocab_list=np.loadtxt(emb_source,dtype=str,delimiter=' ',usecols=[0])
# vocab_emb=np.loadtxt(emb_source,delimiter=' ',usecols=range(1,101))
# print len(vocab_list)#,len(vocab_emb)
# print ','.join(vocab_list[:30])
ff=open(emb_source,'r').readlines()
for i in ff[:50]:
    np_emb=np.zeros(100)
    partindex=i.find(' ')
    word=i[:partindex]
    embedding=map(float,i[(partindex+1):-2].split(' '))
    np_emb[:]=embedding
    vocab_dict[word]=np_emb
print vocab_dict