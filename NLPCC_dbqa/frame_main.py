#coding=utf8
import sys
import time
import re
import pickle
import jieba
import jieba.posseg as pseg
import jieba.analyse
import numpy as np
from numpy import *
import xgboost as xgb
import scipy.spatial.distance as dist
from tqdm import tqdm

def postag(sent):
    result=[]
    words =pseg.cut(sent)
    for w in words:
        result.append((w.word,w.flag))
    return result

def fliter_line(lines):
    lines=lines.replace('请问','')
    lines=lines.replace('我想知道','')
    lines=lines.replace('我很好奇','')
    lines=lines.replace('你知道','')
    lines=lines.replace('谁知道','')
    return lines

def find_lcs_len(s1, s2):
    s1=s1.decode('utf8')
    s2=s2.decode('utf8')
    m = [[ 0 for x in s2] for y in s1]
    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:
                if p1 == 0 or p2 == 0:
                    m[p1][p2] = 1
                else:
                    m[p1][p2] = m[p1-1][p2-1]+1
            elif m[p1-1][p2] < m[p1][p2-1]:
                m[p1][p2] = m[p1][p2-1]
            else:               # m[p1][p2-1] < m[p1-1][p2]
                m[p1][p2] = m[p1-1][p2]
    return m[-1][-1]

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

    def word_overlap_rela(lines):
        dis_numpy=np.zeros([len(lines),1])
        relawords_dict=open('relative_words.txt').read()
        # for idx,line in tqdm(enumerate(lines)):
        for idx,line in enumerate(lines):
            # print '\n'
            # print line.strip()
            each=line.split('\t')
            question,answer=each[0],each[1]
            question=jieba._lcut(question)
            answer=jieba._lcut(answer)

            result=0
            question_relalist=[]
            relalist=relawords_dict.split('\n')
            for que in question:
                for one in relalist:
                    wordslist=one.split(' ')[1:]
                    if que.encode('utf8') in wordslist:
                        question_relalist+=wordslist
            # print ' '.join(question_relalist)

            for ans in answer:
                if ans.encode('utf8') in question_relalist:
                    result+=1
                    # print ans

            dis_numpy[idx,0]=result
        del relawords_dict
        print dis_numpy.shape
        pickle.dump(dis_numpy,open('rela_overlap.np.{}'.format(time.ctime()),'w'))
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

    def max_common(lines):
        dis_numpy=np.zeros([len(lines),1])
        for idx,line in enumerate(lines):
            each=line.split('\t')
            question,answer=each[0],each[1]
            dis_numpy[idx,0]=find_lcs_len(question,answer)
        print dis_numpy.shape
        return dis_numpy

    def parse_keywords(lines):
        dis_numpy=np.zeros([len(lines),1])
        for idx,line in enumerate(lines):
            each=line.split('\t')
            question,answer=each[0],each[1]
            dis_numpy[idx,0]=find_lcs_len(question,answer)
        print dis_numpy.shape
        return dis_numpy

    lines=[fliter_line(line) for line in lines]
    total_featurelist=[]
    total_featurelist.append(word_overlap(lines))
    total_featurelist.append(max_common(lines))
    # total_featurelist.append(word_overlap_rela(lines))
    # total_featurelist.append(topwords_similarity(lines))
    total_featurelist.append(word2vec_cos(lines))
    total_featurelist.append(word2vec_dis(lines))
    total_featurelist.append(word2vec_disall(lines))
    # total_featurelist.append(word2vec_disall_2(lines))

    return total_featurelist

def features_builder_passage(split_idx,lines):
    def ques_parser(question,ques_pos,answers):
        '''answers是list'''
        '''answers是list'''
        def tf_idf(keyword,line,passage=answers):
            '''这里留下了一个隐患，具体就是计算idf的时候，是否要把目标行去掉'''
            keyword=keyword.encode('utf8')
            passage=''.join([each for each in passage if each!=line])
            tf=len(re.findall(keyword,line))
            df=len(re.findall(keyword,passage))
            idf=1.0/(df+1)
            print tf,idf
            return tf*idf

        num_answers=len(answers)
        length=len(question.decode('utf8'))
        dis_1,dis_2,dis_3=[],[],[]
        if '什么' in question:
            aim_idx=[length,length]
            for idx,word in enumerate(ques_pos):
                if '什么'.decode('utf8') in word[0]:
                    aim_idx=[idx,idx]
                    break
            if ques_pos[aim_idx[0]-1][0]=='是'.decode('utf8'):
                aim_idx[0]=aim_idx[0]-1

            pos_aim=[(i[0],i[1],idx) for idx,i in enumerate(ques_pos) if ('n' in i[1] or 'v' in i[1])]

            for aim in pos_aim:
                if aim[2]-aim_idx[0]==-1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[1]==1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[0]==-2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[1]==2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[0]==-3:
                    dis_3.append(aim)
                if aim[2]-aim_idx[1]==3:
                    dis_3.append(aim)
            pass
        elif '谁' in question:
            aim_idx=[length,length]
            for idx,word in enumerate(ques_pos):
                if '谁'.decode('utf8') in word[0]:
                    aim_idx=[idx,idx]
                    break
            if ques_pos[aim_idx[0]-1][0]=='是'.decode('utf8'):
                aim_idx[0]=aim_idx[0]-1

            pos_aim=[(i[0],i[1],idx) for idx,i in enumerate(ques_pos) if ('n' in i[1] or 'v' in i[1])]

            for aim in pos_aim:
                if aim[2]-aim_idx[0]==-1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[1]==1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[0]==-2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[1]==2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[0]==-3:
                    dis_3.append(aim)
                if aim[2]-aim_idx[1]==3:
                    dis_3.append(aim)
            pass
        elif '哪' in question:
            aim_idx=[length,length]
            for idx,word in enumerate(ques_pos):
                if '哪'.decode('utf8') in word[0]:
                    aim_idx=[idx,idx]
                    break
            if ques_pos[aim_idx[0]-1][0]=='是'.decode('utf8'):
                aim_idx[0]=aim_idx[0]-1

            pos_aim=[(i[0],i[1],idx) for idx,i in enumerate(ques_pos) if ('n' in i[1] or 'v' in i[1])]

            for aim in pos_aim:
                if aim[2]-aim_idx[0]==-1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[1]==1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[0]==-2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[1]==2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[0]==-3:
                    dis_3.append(aim)
                if aim[2]-aim_idx[1]==3:
                    dis_3.append(aim)
            pass
        elif '几' in question:
            aim_idx=[length,length]
            for idx,word in enumerate(ques_pos):
                if '几'.decode('utf8') in word[0]:
                    aim_idx=[idx,idx]
                    break
            if ques_pos[aim_idx[0]-1][0]=='是'.decode('utf8'):
                aim_idx[0]=aim_idx[0]-1

            pos_aim=[(i[0],i[1],idx) for idx,i in enumerate(ques_pos) if ('n' in i[1] or 'v' in i[1])]

            for aim in pos_aim:
                if aim[2]-aim_idx[0]==-1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[1]==1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[0]==-2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[1]==2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[0]==-3:
                    dis_3.append(aim)
                if aim[2]-aim_idx[1]==3:
                    dis_3.append(aim)
            pass
        elif '多少' in question:
            aim_idx=[length,length]
            for idx,word in enumerate(ques_pos):
                if '多少'.decode('utf8') in word[0]:
                    aim_idx=[idx,idx]
                    break
            if ques_pos[aim_idx[0]-1][0]=='是'.decode('utf8'):
                aim_idx[0]=aim_idx[0]-1

            pos_aim=[(i[0],i[1],idx) for idx,i in enumerate(ques_pos) if ('n' in i[1] or 'v' in i[1])]

            for aim in pos_aim:
                if aim[2]-aim_idx[0]==-1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[1]==1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[0]==-2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[1]==2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[0]==-3:
                    dis_3.append(aim)
                if aim[2]-aim_idx[1]==3:
                    dis_3.append(aim)
            pass
        elif '哪' in question:
            aim_idx=[length,length]
            for idx,word in enumerate(ques_pos):
                if '哪'.decode('utf8') in word[0]:
                    aim_idx=[idx,idx]
                    break
            if ques_pos[aim_idx[0]-1][0]=='是'.decode('utf8'):
                aim_idx[0]=aim_idx[0]-1

            pos_aim=[(i[0],i[1],idx) for idx,i in enumerate(ques_pos) if ('n' in i[1] or 'v' in i[1])]

            for aim in pos_aim:
                if aim[2]-aim_idx[0]==-1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[1]==1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[0]==-2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[1]==2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[0]==-3:
                    dis_3.append(aim)
                if aim[2]-aim_idx[1]==3:
                    dis_3.append(aim)
            pass
        elif '怎么' in question:
            aim_idx=[length,length]
            for idx,word in enumerate(ques_pos):
                if '怎么'.decode('utf8') in word[0]:
                    aim_idx=[idx,idx]
                    break
            if ques_pos[aim_idx[0]-1][0]=='是'.decode('utf8'):
                aim_idx[0]=aim_idx[0]-1

            pos_aim=[(i[0],i[1],idx) for idx,i in enumerate(ques_pos) if ('n' in i[1] or 'v' in i[1])]

            for aim in pos_aim:
                if aim[2]-aim_idx[0]==-1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[1]==1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[0]==-2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[1]==2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[0]==-3:
                    dis_3.append(aim)
                if aim[2]-aim_idx[1]==3:
                    dis_3.append(aim)
            pass
        elif re.findall('多[重厚快深宽薄大高远长久]',question):
            aim_idx=[length,length]
            for idx,word in enumerate(ques_pos):
                if '多'.decode('utf8') in word[0]:
                    aim_idx=[idx,idx]
                    break
            if ques_pos[aim_idx[0]-1][0]=='是'.decode('utf8'):
                aim_idx[0]=aim_idx[0]-1

            pos_aim=[(i[0],i[1],idx) for idx,i in enumerate(ques_pos) if ('n' in i[1] or 'v' in i[1])]

            for aim in pos_aim:
                if aim[2]-aim_idx[0]==-1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[1]==1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[0]==-2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[1]==2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[0]==-3:
                    dis_3.append(aim)
                if aim[2]-aim_idx[1]==3:
                    dis_3.append(aim)
            pass
        elif '如何' in question:
            aim_idx=[length,length]
            for idx,word in enumerate(ques_pos):
                if '如何'.decode('utf8') in word[0]:
                    aim_idx=[idx,idx]
                    break
            if ques_pos[aim_idx[0]-1][0]=='是'.decode('utf8'):
                aim_idx[0]=aim_idx[0]-1

            pos_aim=[(i[0],i[1],idx) for idx,i in enumerate(ques_pos) if ('n' in i[1] or 'v' in i[1])]

            for aim in pos_aim:
                if aim[2]-aim_idx[0]==-1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[1]==1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[0]==-2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[1]==2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[0]==-3:
                    dis_3.append(aim)
                if aim[2]-aim_idx[1]==3:
                    dis_3.append(aim)
            pass
        elif '啥' in question:
            aim_idx=[length,length]
            for idx,word in enumerate(ques_pos):
                if '啥'.decode('utf8') in word[0]:
                    aim_idx=[idx,idx]
                    break
            if ques_pos[aim_idx[0]-1][0]=='是'.decode('utf8'):
                aim_idx[0]=aim_idx[0]-1

            pos_aim=[(i[0],i[1],idx) for idx,i in enumerate(ques_pos) if ('n' in i[1] or 'v' in i[1])]

            for aim in pos_aim:
                if aim[2]-aim_idx[0]==-1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[1]==1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[0]==-2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[1]==2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[0]==-3:
                    dis_3.append(aim)
                if aim[2]-aim_idx[1]==3:
                    dis_3.append(aim)
            pass
        elif '怎样' in question:
            aim_idx=[length,length]
            for idx,word in enumerate(ques_pos):
                if '怎样'.decode('utf8') in word[0]:
                    aim_idx=[idx,idx]
                    break
            if ques_pos[aim_idx[0]-1][0]=='是'.decode('utf8'):
                aim_idx[0]=aim_idx[0]-1

            pos_aim=[(i[0],i[1],idx) for idx,i in enumerate(ques_pos) if ('n' in i[1] or 'v' in i[1])]

            for aim in pos_aim:
                if aim[2]-aim_idx[0]==-1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[1]==1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[0]==-2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[1]==2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[0]==-3:
                    dis_3.append(aim)
                if aim[2]-aim_idx[1]==3:
                    dis_3.append(aim)
            pass
        elif '时间？' in question:
            aim_idx=[length,length]
            for idx,word in enumerate(ques_pos):
                if '时间？'.decode('utf8') in word[0]:
                    aim_idx=[idx,idx]
                    break
            if ques_pos[aim_idx[0]-1][0]=='是'.decode('utf8'):
                aim_idx[0]=aim_idx[0]-1

            pos_aim=[(i[0],i[1],idx) for idx,i in enumerate(ques_pos) if ('n' in i[1] or 'v' in i[1])]

            for aim in pos_aim:
                if aim[2]-aim_idx[0]==-1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[1]==1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[0]==-2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[1]==2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[0]==-3:
                    dis_3.append(aim)
                if aim[2]-aim_idx[1]==3:
                    dis_3.append(aim)
            pass
        else:
            '''最后这种情况应该就是最后直接带一个问号的'''
            print question
            aim_idx=[length,length]
            pos_aim=[(i[0],i[1],idx) for idx,i in enumerate(ques_pos) if ('n' in i[1] or 'v' in i[1])]
            for aim in pos_aim:
                if aim[2]-aim_idx[0]==-1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[1]==1:
                    dis_1.append(aim)
                if aim[2]-aim_idx[0]==-2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[1]==2:
                    dis_2.append(aim)
                if aim[2]-aim_idx[0]==-3:
                    dis_3.append(aim)
                if aim[2]-aim_idx[1]==3:
                    dis_3.append(aim)

        dis_numpy=np.zeros([num_answers,3])
        for idx,line in enumerate(answers):
            for i in dis_1:
                dis_numpy[idx,0]+=tf_idf(i[0],line)
            for i in dis_2:
                dis_numpy[idx,1]+=tf_idf(i[0],line)
            for i in dis_3:
                dis_numpy[idx,2]+=tf_idf(i[0],line)
        return dis_numpy

    split_idx.append(len(lines))
    que_list=[]
    ans_list=[]
    for i in range(len(split_idx)):
        try:
            question=lines[split_idx[i]].split('\t')[0]
            one_answer_list=[line.split('\t')[1]  for line in lines[split_idx[i]:split_idx[i+1]]]
            que_list.append(question)
            ans_list.append(one_answer_list)
        except IndexError:
            pass

    assert len(que_list)==len(ans_list)
    dis_numpy_list=[]
    for question,ansers in zip(que_list,ans_list):
        question=fliter_line(question)
        ques_pos=postag(question)
        dis_numpy_list.append(ques_parser(question,ques_pos,ansers))

    final_dis_numpy=np.concatenate(dis_numpy_list,axis=0)
    return final_dis_numpy


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

def cal_main(train_file,test_file,score_file,train_target=None,test_target=None):
    if train_target:
        train_lable=np.zeros(len(train_target))
        train_lable[:]=map(int,train_target)
        dtrain = xgb.DMatrix(train_file,train_lable)
    else:
        dtrain = xgb.DMatrix(train_file)
    print 'dtrain finished.'
    if test_target:
        test_lable=np.zeros(len(test_target))
        test_lable[:]=map(int,test_target)
        dtest = xgb.DMatrix(test_file,test_lable)
    else:
        dtest = xgb.DMatrix(test_file)
    print 'dtest finished.'
    # specify parameters via map
    evallist  = [(dtest,'eval'), (dtrain,'train')]
    param = {'booster':'gbtree',
             'max_depth':7,
             'eta':0.06,
             'min_child_weight':30,
             'subsample':1,
             'silent':0,
             'objective':'binary:logistic',
             # 'objective':'reg:linear',
             'lambda':0.3,
             'alpha':0.2}
    num_round = 100
    bst = xgb.train(param, dtrain, num_round ,evallist)
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
    score_file='results/result_0623_cover&w2v&dists'
    construct=10

    if construct:
        build_vocab=False
        # if build_vocab:
        #     vocab=get_vocab(train_file,test_file)
        # else:
        #     vocab=pickle.load(open('vocabSet_in_NLPCC_main'))
        train_split_idx,train_ansList,train_lines=construct_train(train_file)
        pickle.dump(train_ansList,open(train_features+'.train_label_np','w'))
        test_split_idx,test_ansList,test_lines=construct_test(test_file)
        pickle.dump(test_ansList,open(test_features+'.test_label_np','w'))
        # del _
        # print train_ansList[0:20]
        print ''.join(train_lines[0:3])
        print ''.join(test_lines[0:3])

        total_featurelist_test=features_builder_passage(test_split_idx,test_lines)
        print 1/0
        total_featurelist_test=features_builder(test_split_idx,test_lines)
        test_np=format_xgboost(total_featurelist_test,out_path=test_features)
        pickle.dump(test_np,open(test_features+'.np','w'))
        print 'test feats finished'

        total_featurelist_train=features_builder(train_split_idx,train_lines)
        train_np=format_xgboost(total_featurelist_train,out_path=train_features,target=train_ansList)
        pickle.dump(train_np,open(train_features+'.np','w'))
        print 'train feats finished'
    else:
        # train_np=pickle.load(open(train_features+'.all_np'))
        train_np=pickle.load(open(train_features+'.np'))
        tmp1=pickle.load(open('rela_overlap.np.train'))
        train_np=np.concatenate((tmp1,train_np),axis=1)
        train_ansList=pickle.load(open(train_features+'.train_label_np'))
        # test_np=pickle.load(open(test_features+'.all_np'))
        test_np=pickle.load(open(test_features+'.np'))
        tmp2=pickle.load(open('rela_overlap.np.test'))
        test_np=np.concatenate((tmp2,test_np),axis=1)
        test_ansList=pickle.load(open(test_features+'.test_label_np'))

        print train_np.shape

    cal_main(train_np,test_np,score_file,train_target=train_ansList,test_target=test_ansList)
    # cal_main(train_features,test_features,score_file)
