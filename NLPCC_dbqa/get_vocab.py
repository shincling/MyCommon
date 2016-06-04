#coding=utf8
import sys
import re
import jieba

input_path='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/nlpcc-iccpol-2016.dbqa.training-data'
f_input=open(input_path,'r').readlines()
print 'total lines of input is {}'.format(len(f_input))
sent_set=set()
question_set=set()
ans_set=set()
for line in f_input:
    each=line.split('\t')
    question_set.add(each[0])
    sent_set.add(each[0])
    sent_set.add(each[1])
    if each[1] in ans_set:
        print each[1]
    ans_set.add(each[1])
print 'total num of sents:{}'.format(len(sent_set))
print len(question_set)
print len(ans_set)

for ans in ans_set:
    if ans in question_set:
        print ans
        break
