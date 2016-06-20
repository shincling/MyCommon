#coding=utf8
import sys
import re
import pickle
import jieba
import jieba.posseg as pseg
import jieba.analyse
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
            question_vec_01[i,0]=1.0
        if word in answer:
            answer_vec_01[i,0]=1.0
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
        dis_numpy=np.zeros([len(lines),6])
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
            # dis_numpy[idx,1]=1-dist.correlation(question_vec_01,answer_vec_01)
            dis_numpy[idx,1]=1-dist.jaccard(question_vec_01,answer_vec_01)
            dis_numpy[idx,2]=1-dist.hamming(question_vec_01,answer_vec_01)
            # dis_numpy[idx,3]=1-dist.correlation(question_vec_01,answer_vec_01)
            # dis_numpy[idx,4]=1-dist.correlation(question_vector,answer_vector)
            # dis_numpy[idx,5]=1-dist.jaccard(question_vec_01,answer_vec_01)
            # dis_numpy[idx,6]=1-dist.hamming(question_vec_01,answer_vec_01)
            # dis_numpy[idx,7]=1-dist.correlation(question_vec_01,answer_vec_01)
            # dis_numpy[idx,4]=dis_numpy[idx,3]
            dis_numpy[idx,3]=1-dist.rogerstanimoto(question_vec_01,answer_vec_01)
            dis_numpy[idx,4]=1-dist.cityblock(question_vec_01,answer_vec_01)
            dis_numpy[idx,5]=1-dist.matching(question_vec_01,answer_vec_01)

        del word_dict
        print dis_numpy.shape
        return dis_numpy

    def word2vec_disall_2(lines):
        dis_numpy=np.zeros([len(lines),3])
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

            # dis_numpy[idx,0]=1-dist.correlation(question_vector,answer_vector)
            # dis_numpy[idx,1]=1-dist.jaccard(question_vec_01,answer_vec_01)
            # dis_numpy[idx,2]=1-dist.hamming(question_vec_01,answer_vec_01)
            # dis_numpy[idx,3]=1-dist.correlation(question_vec_01,answer_vec_01)
            # dis_numpy[idx,4]=1-dist.correlation(question_vector,answer_vector)
            # dis_numpy[idx,5]=1-dist.jaccard(question_vec_01,answer_vec_01)
            # dis_numpy[idx,6]=1-dist.hamming(question_vec_01,answer_vec_01)
            # dis_numpy[idx,7]=1-dist.correlation(question_vec_01,answer_vec_01)
            # dis_numpy[idx,4]=dis_numpy[idx,3]
            dis_numpy[idx,0]=1-dist.rogerstanimoto(question_vec_01,answer_vec_01)
            dis_numpy[idx,1]=1-dist.cityblock(question_vec_01,answer_vec_01)
            dis_numpy[idx,2]=1-dist.matching(question_vec_01,answer_vec_01)

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
        print dis_numpy.shape
        return dis_numpy

    def topwords_similarity(lines):
        dis_numpy=np.zeros([len(lines),4])
        for idx,line in enumerate(lines):
            each=line.split('\t')
            question,answer=each[0],each[1]
            question=jieba.analyse.extract_tags(question,5)
            answer=jieba.analyse.extract_tags(answer,5)

            result=0
            for que in question:
                for ans in answer:
                    if ans==que:
                        result+=1
            dis_numpy[idx,0]=result

            question_vec,answer_vec=binary_twosent(question,answer)
            dis_numpy[idx,1]=1-dist.jaccard(question_vec,answer_vec)
            dis_numpy[idx,2]=1-dist.hamming(question_vec,answer_vec)
            dis_numpy[idx,3]=1-dist.cosine(question_vec,answer_vec)
        print dis_numpy.shape
        return dis_numpy
    total_featurelist=[]
    total_featurelist.append(word_overlap(lines))
    total_featurelist.append(topwords_similarity(lines))
    total_featurelist.append(word2vec_cos(lines))
    total_featurelist.append(word2vec_dis(lines))
    total_featurelist.append(word2vec_disall(lines))
    total_featurelist.append(word2vec_disall_2(lines))

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
    return final_feature

def cal_main(train_file,test_file,score_file,train_target=None):
    if train_target:
        train_lable=np.zeros(len(train_target))
        train_lable[:]=map(int,train_target)
        dtrain = xgb.DMatrix(train_file,train_lable)
    else:
        dtrain = xgb.DMatrix(train_file)
    print 'dtrain finished.'
    dtest = xgb.DMatrix(test_file)
    print 'dtest finished.'
    # specify parameters via map
    param = {'booster':'gbtree',
             'max_depth':7,
             'eta':0.02,
             'min_child_weight':5,
             'subsample':1,
             'silent':0,
             'objective':'binary:logistic',
             'lambda':0.3,
             'alpha':0.2}
    num_round = 300
    bst = xgb.train(param, dtrain, num_round)
    # make prediction
    train_score=bst.predict(dtrain)
    preds = bst.predict(dtest)
    print bst.get_fscore()
    open(score_file+'_valid','w').write('\r\n'.join([str(i) for i in preds]))
    open(score_file+'_train','w').write('\r\n'.join([str(i) for i in train_score]))
    # return preds

if __name__=='__main__':
    train_file='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/data_valid/train7_1'
    test_file='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/data_valid/valid3_1'
    # train_file='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/data_valid/train_demo'
    # test_file='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/data_valid/valid_demo'
    train_features='results/train_ssss.txt'
    test_features='results/test_ssss.txt'
    # train_features='/home/shin/XGBoost/xgboost/demo/binary_classification/agaricus.txt.train'
    # test_features='/home/shin/XGBoost/xgboost/demo/binary_classification/agaricus.txt.test'
    score_file='results/result_0620_cover&w2v&dists'
    construct=0

    if construct:
        build_vocab=False
        if build_vocab:
            vocab=get_vocab(train_file,test_file)
        else:
            vocab=pickle.load(open('vocabSet_in_NLPCC_main'))
        train_split_idx,train_ansList,train_lines=construct_train(train_file)
        pickle.dump(train_ansList,open(train_features+'.label_np','w'))
        test_split_idx,_,test_lines=construct_test(test_file)
        del _
        # print train_ansList[0:20]
        print ''.join(train_lines[0:3])
        print ''.join(test_lines[0:3])
        total_featurelist_train=features_builder(train_split_idx,train_lines)
        train_np=format_xgboost(total_featurelist_train,out_path=train_features,target=train_ansList)
        total_featurelist_test=features_builder(test_split_idx,test_lines)
        test_np=format_xgboost(total_featurelist_test,out_path=test_features)
        pickle.dump(train_np,open(train_features+'.np','w'))
        pickle.dump(test_np,open(test_features+'.np','w'))
    else:
        train_np=pickle.load(open(train_features+'.np'))
        train_ansList=pickle.load(open(train_features+'.label_np'))
        test_np=pickle.load(open(test_features+'.np'))

    cal_main(train_np,test_np,score_file,train_target=train_ansList)
    # cal_main(train_features,test_features,score_file)
