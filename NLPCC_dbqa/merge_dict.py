#coding=utf8
import pickle
from tqdm import tqdm
import numpy as np

vocab_NLPCC_0701=pickle.load(open('vocabSet_in_NLPCC_0701'))
vocab_NLPCC_0605=pickle.load(open('vocabSet_in_NLPCC'))

print len(vocab_NLPCC_0605),len(vocab_NLPCC_0701)

vocab_NLPCC_0605.update(vocab_NLPCC_0701)

print len(vocab_NLPCC_0605)

pickle.dump(vocab_NLPCC_0605,open('vocabSet_NLPCC_all','w'))


