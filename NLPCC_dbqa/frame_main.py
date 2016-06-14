#coding=utf8
import sys
import re
import pickle
import jieba
from numpy import *

def cos_dis(vector1,vector2):
    return dot(vector1,vector2)/(linalg.norm(vector1)*linalg.norm(vector2))

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

def construct_total_input(input_path):
    f_input_train=open(train_file,'r').readlines()
    f_input_test=open(test_file,'r').readlines()
    f_input=f_input_train+f_input_test
    print 'total lines of input is {}'.format(len(f_input))
    new_question_indexList=[]
    ans_indexList=[]
    result_list=[]
    old_question=''
    for idx,question in enumerate(f_input):
        spl=question.strip().split('\t')
        now_quesiton,now_ans=spl[0],spl[-1]
        if now_quesiton!=old_question:
            new_question_indexList.append(idx)
            old_question=now_quesiton
        if now_ans=='1':
            ans_indexList.append(idx)
        result_list.append(spl[-1])
    print 'total num of questions is :{}'.format(len(new_question_indexList))
    print 'total num of ans=1 list is {}'.format(len(ans_indexList))
    return new_question_indexList,ans_indexList,f_input

def construct_train(input_path):
    f_input=open(input_path,'r').readlines()
    print 'total lines of train file is {}'.format(len(f_input))
    new_question_indexList=[]
    ans_indexList=[]
    result_list=[]
    old_question=''
    for idx,question in enumerate(f_input):
        spl=question.strip().split('\t')
        now_quesiton,now_ans=spl[0],spl[2]
        if now_quesiton!=old_question:
            new_question_indexList.append(idx)
            old_question=now_quesiton
        if now_ans=='1' or now_ans=='0':
            ans_indexList.append(now_ans)
        result_list.append(spl[-1])
    print 'total num of questions is :{}'.format(len(new_question_indexList))
    print 'total num of ans is :{}  (Ps: it should be equel with the train file lines num)\n'.format(len(ans_indexList))
    return new_question_indexList,ans_indexList,f_input

def construct_test(input_path):
    f_input=open(input_path,'r').readlines()
    print 'total lines of test file is {}'.format(len(f_input))
    new_question_indexList=[]
    ans_indexList=[]
    result_list=[]
    old_question=''
    for idx,question in enumerate(f_input):
        spl=question.strip().split('\t')
        now_quesiton=spl[0]
        if now_quesiton!=old_question:
            new_question_indexList.append(idx)
            old_question=now_quesiton
        result_list.append(spl[-1])
    print 'total num of questions is :{}\n'.format(len(new_question_indexList))
    return new_question_indexList,ans_indexList,f_input

def features_builder(split_idx,lines):
    def word2vec_cos(lines):
        dis_list=[]
        word_dict=pickle.load(open('nlpcc_dict_20160605'))
        for line in lines:
            each=line.split('\t')
            question=jieba._lcut(each[0])
            question_vector=zeros(100)
            for word in question:
                one_vec=word_dict[word.encode('utf8')]
                question_vector+=one_vec

            answer=jieba._lcut(each[1])
            answer_vector=zeros(100)
            for word in answer:
                one_vec=word_dict[word.encode('utf8')]
                answer_vector+=one_vec

            dis=cos_dis(question_vector,answer_vector)
            dis_list.append(dis)
        del word_dict
        return dis_list

    def format_xgboost(total_features):
        length=set()
        for feature in total_features:
            pass

    total_featurelist=[]
    total_featurelist.append(word2vec_cos(lines))


    return format_xgboost(total_featurelist)


if __name__=='__main__':
    train_file='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/data_valid/train7_1'
    test_file='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/data_valid/valid3_1'
    build_vocab=False
    if build_vocab:
        vocab=get_vocab(train_file,test_file)
    else:
        vocab=pickle.load(open('vocabSet_in_NLPCC_main'))
    train_split_idx,train_ansList,train_lines=construct_train(train_file)
    test_split_idx,_,test_lines=construct_test(test_file)
    del _
    # print train_ansList[0:20]
    print ''.join(train_lines[0:3])
    print ''.join(test_lines[0:3])


