# -*- coding: utf8 -*-
from numpy import *
from tqdm import tqdm
import pickle
import jieba


def cos_dis(vector1,vector2):
    return dot(vector1,vector2)/(linalg.norm(vector1)*linalg.norm(vector2))

word_dict=pickle.load(open('nlpcc_dict_20160605'))
input_path='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/nlpcc-iccpol-2016.dbqa.training-data'
out_path='similarity_0606'
f_input=open(input_path,'r').readlines()
print 'total lines of input is {}'.format(len(f_input))
dis_list=[]
for line in tqdm(f_input):
    each=line.split('\t')
    question=jieba._lcut(each[0])
    question_vector=zeros(100)
    # print question_vector.shape
    for word in question:
        one_vec=word_dict[word.encode('utf8')]
        # print one_vec.shape
        question_vector+=one_vec

    answer=jieba._lcut(each[1])
    answer_vector=zeros(100)
    for word in answer:
        one_vec=word_dict[word.encode('utf8')]
        answer_vector+=one_vec

    dis=cos_dis(question_vector,answer_vector)
    dis_list.append(dis)
print len(dis_list)
print dis_list[:30]

f_out=open(out_path,'w')
for dis in dis_list:
    f_out.write(dis+'\r\n')
