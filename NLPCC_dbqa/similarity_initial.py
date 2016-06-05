# -*- coding: utf8 -*-
from numpy import *
import numpy as np
import pickle
import jieba

def cos_dis(vector1,vector2):
    return dot(vector1,vector2)/(linalg.norm(vector1)*linalg.norm(vector2))

word_dict=pickle.load('nlpcc_dict_20160605')
input_path='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/nlpcc-iccpol-2016.dbqa.training-data'
f_input=open(input_path,'r').readlines()
print 'total lines of input is {}'.format(len(f_input))
dis_list=[]
for line in f_input:
    each=line.split('\t')
    question=jieba._lcut(each[1])
    question_vector=zeros(100)
    for word in question:
        one_vec=word_dict[word]
        question_vector+=one_vec

    answer=jieba._lcut(each[1])
    answer_vector=zeros(100)
    for word in answer:
        one_vec=word_dict[word]
        answer_vector+=one_vec

    dis=cos_dis(question_vector,answer_vector)
    dis_list.append(dis)
print len(dis_list)
print dis_list[:30]

