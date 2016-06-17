#coding=utf8
import sys
import re
import pickle
import jieba
import numpy as np
import xgboost as xgb
import scipy.spatial.distance as dist

def cos_dis(vector1,vector2):
    return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

def binary_twosent(question,answer):
    all_words=set(question+answer)
    question_vec_01=np.zeros([len(all_words),1])
    answer_vec_01=np.zeros([len(all_words),1])
    for i in range(len(all_words)):
        word=list(all_words)[i]
        if word in question:
            question_vec_01[i,0]=1
        if word in answer:
            answer_vec_01[i,0]=1
    return question_vec_01,answer_vec_01

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
    return new_question_indexList,result_list,f_input

def features_builder(split_idx,lines):
    def word2vec_cos(lines):
        dis_numpy=np.zeros([len(lines),1])
        word_dict=pickle.load(open('nlpcc_dict_20160605'))
        for idx,line in enumerate(lines):
            each=line.split('\t')
            question=jieba._lcut(each[0])
            question_vector=np.zeros(100)
            for word in question:
                try:
                    one_vec=word_dict[word.encode('utf8')]
                except  KeyError:
                    one_vec=np.random.normal(size=(100))
                question_vector+=one_vec

            answer=jieba._lcut(each[1])
            answer_vector=np.zeros(100)
            for word in answer:
                try:
                    one_vec=word_dict[word.encode('utf8')]
                except KeyError:
                    one_vec=np.random.normal(size=(100))
                answer_vector+=one_vec

            dis=cos_dis(question_vector,answer_vector)
            dis_numpy[idx,0]=dis
        del word_dict
        print dis_numpy.shape
        return dis_numpy

    def word2vec_dis(lines):
        dis_numpy=np.zeros([len(lines),1])
        word_dict=pickle.load(open('nlpcc_dict_20160605'))
        for idx,line in enumerate(lines):
            each=line.split('\t')
            question=jieba._lcut(each[0])
            question_vector=np.zeros(100)
            for word in question:
                try:
                    one_vec=word_dict[word.encode('utf8')]
                except  KeyError:
                    one_vec=np.random.normal(size=(100))
                question_vector+=one_vec

            answer=jieba._lcut(each[1])
            answer_vector=np.zeros(100)
            for word in answer:
                try:
                    one_vec=word_dict[word.encode('utf8')]
                except KeyError:
                    one_vec=np.random.normal(size=(100))
                answer_vector+=one_vec

            # dis=cos_dis(question_vector,answer_vector)
            dis = np.linalg.norm(question_vector-answer_vector)
            dis_numpy[idx,0]=dis
        del word_dict
        print dis_numpy.shape
        return dis_numpy

    def word2vec_disall(lines):
        dis_numpy=np.zeros([len(lines),4])
        word_dict=pickle.load(open('nlpcc_dict_20160605'))
        for idx,line in enumerate(lines):
            each=line.split('\t')
            question=jieba._lcut(each[0])
            question_vector=np.zeros(100)
            for word in question:
                try:
                    one_vec=word_dict[word.encode('utf8')]
                except  KeyError:
                    one_vec=np.random.normal(size=(100))
                question_vector+=one_vec

            answer=jieba._lcut(each[1])
            answer_vector=np.zeros(100)
            for word in answer:
                try:
                    one_vec=word_dict[word.encode('utf8')]
                except KeyError:
                    one_vec=np.random.normal(size=(100))
                answer_vector+=one_vec

            question_vec_01,answer_vec_01=binary_twosent(question,answer)

            dis_numpy[idx,0]=1-dist.correlation(question_vector,answer_vector)
            dis_numpy[idx,1]=1-dist.jaccard(question_vec_01,answer_vec_01)
            dis_numpy[idx,2]=1-dist.hamming(question_vec_01,answer_vec_01)
            dis_numpy[idx,3]=1-dist.correlation(question_vec_01,answer_vec_01)

        del word_dict
        print dis_numpy.shape
        return dis_numpy

    def word_overlap(lines):
        dis_numpy=np.zeros([len(lines),1])
        for idx,line in enumerate(lines):
            each=line.split('\t')
            question,answer=each[0],each[1]
            question=jieba._lcut(question)
            answer=jieba._lcut(answer)

            result=0
            for que in question:
                for ans in answer:
                    if ans==que:
                        result+=1
            dis_numpy[idx,0]=result
        return dis_numpy

    total_featurelist=[]
    total_featurelist.append(word_overlap(lines))
    total_featurelist.append(word2vec_cos(lines))
    total_featurelist.append(word2vec_dis(lines))
    total_featurelist.append(word2vec_disall(lines))

    return total_featurelist

def format_xgboost(total_features,out_path,target=None):
    final_feature=np.concatenate(total_features,axis=1)
    feature_num=final_feature.shape[1]

    all_lines=''
    for hang in range(final_feature.shape[0]):
        line=''
        for lie in range(1,feature_num+1):
            line+='{}:{} '.format(lie,final_feature[hang,lie-1])
        line=line.strip()+'\n'
        all_lines+=line

    if target:
        tmp=''
        lines_list=all_lines.splitlines()
        assert len(target)==len(lines_list)
        print 'len target = lines_list , {}'.format(len(target))

        for tar,line in zip(target,lines_list):
            new_line=str(tar)+' '+line+'\n'
            tmp+=new_line
        all_lines=tmp

    open(out_path,'w').write(all_lines.strip())

def cal_main(train_file,test_file,score_file):
    dtrain = xgb.DMatrix(train_file)
    dtest = xgb.DMatrix(test_file)
    # specify parameters via map
    param = {'booster':'gblinear',
             'max_depth':50,
             'eta':0.3,
             'min_child_weight':10,
             'subsample':1,
             'silent':1,
             'objective':'binary:logistic',
             'lambda':0.,
             'alpha':0.}
    num_round = 100
    bst = xgb.train(param, dtrain, num_round)
    # make prediction
    train_score=bst.predict(dtrain)
    preds = bst.predict(dtest)
    print bst.get_fscore()
    open(score_file,'w').write('\r\n'.join([str(i) for i in preds]))
    open(score_file+'_train','w').write('\r\n'.join([str(i) for i in train_score]))
    # return preds

if __name__=='__main__':
    train_file='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/data_valid/train7_1'
    test_file='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/data_valid/valid3_1'
    train_features='results/train_ssss.txt'
    test_features='results/test_ssss.txt'
    score_file='results/result_0617_cover&w2v&dists'
    construct=0

    if construct:
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
        total_featurelist_train=features_builder(train_split_idx,train_lines)
        # total_featurelist_test=features_builder(test_split_idx,test_lines)
        format_xgboost(total_featurelist_train,out_path=train_features,target=train_ansList)
        # format_xgboost(total_featurelist_test,out_path=test_features)

    cal_main(train_features,test_features,score_file)
