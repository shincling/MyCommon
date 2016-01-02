# -*- coding: utf-8 -*-

import ujson
import re
import os


path='/home/shin/Memect/ntdb/data/result/1229/logs_1229'
path='/home/shin/Memect/ntdb/data/result/0102/logs_0102'
files=[]
len_0=0
len_big=0
len_count=0
aa=0
errorInfinal=0
for file in os.listdir(path):
    filename=path+'/'+file
    print filename
    files.append(filename)
    f=open(filename,'r')
    content=f.read()
    if 'total' in content:
        if content[-2]=='0':
            len_0+=1
        else:
            len_big+=1
        if int(content[0:-1].split(":")[-1])>10:
            aa+=1
        len_count+=1
    else:
        errorInfinal+=1

print 'Len=0: %d'%len_0
print 'Len>0: %d'%len_big
print 'hasLen: %d'%len_count
print 'noLen: %d'%errorInfinal
print 'Len>10:%d'%aa

print float(aa)/len_count
