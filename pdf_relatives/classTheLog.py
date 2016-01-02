# -*- coding: utf-8 -*-

import ujson
import re
import os


path='/home/shin/Memect/ntdb/data/result/1229/logs_1229'
files=[]
len_0=0
len_big=0
len_count=0
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
        len_count+=1
    else:
        errorInfinal+=1

print len_0,len_big,len_count,errorInfinal

