# -*- coding: utf-8 -*-

import ujson
import re
import os


path='/home/shin/Memect/ntdb/data/result/1229/logs_1229'
path='/home/shin/Memect/ntdb/data/result/0103_2/logs_0103_2'
files=[]
len_0=0
len_big=0
len_count=0
aa=0
errorInfinal=0
bb=0
cc=0
for file in os.listdir(path):
    filename=path+'/'+file
    # print filename
    files.append(filename)
    f=open(filename,'r')
    content=f.read()
    if 'total' in content:
        if content[-2]=='0':
            len_0+=1

        else:
            len_big+=1
            ff=open('/home/shin/Memect/ntdb/data/result/0103_2/all/'+file[:-4]+'_Table1219_word_all.json','r')
            all=ff.read()
            page_total=len(re.findall('"\d+?":',all))
            if page_total==1:
                bb+=1
                # print file,re.findall('"\d+?":',all)[0]
            if page_total>1:
                cc+=1

        if int(content[0:-1].split(":")[-1])>10:
            aa+=1
        len_count+=1
    else:
        errorInfinal+=1
        print file

print 'Len=0: %d'%len_0
print 'Len>0: %d'%len_big
print 'hasLen: %d'%len_count
print 'noLen: %d'%errorInfinal
print 'Len>10:%d'%aa

print float(aa)/len_count

print bb
print cc
'''
Len=0: 54
Len>0: 316
hasLen: 370
noLen: 291
Len>10:125
0.337837837838
233
83
'''

'''  table_0103
Len=0: 28
Len>0: 225
hasLen: 253
noLen: 284
Len>10:89
0.351778656126
162
63
'''

''' table_0103_2
Len=0: 25
Len>0: 179
hasLen: 204
noLen: 300
Len>10:67
0.328431372549
118
61
'''