# -*- coding: utf-8 -*-
from elasticsearch import Elasticsearch
import ujson
import re
import os

es = Elasticsearch('127.0.0.1:9200')

path='/home/shin/MyGit/Common/MyCommon/jsonData'
files=[]
for file in os.listdir(path):
    filename=path+'/'+file
    print filename
    files.append(filename)

    obj = ujson.load(open(filename, 'r'))
    if True:
        es.index('shin_0', 'shin_1', obj, id=re.findall(r'\d+',filename))
        es.index
        #print es
        es.get('shin_0','430002')


        pass


    else:
        es.index('shin_0', 'shin_1', obj)


res = es.search(
index='shin_0',
doc_type='shin_1',
body={
    'query': {
        'match':{
            'sourceWeb':'touziren'
        }


      }
    }
)

print res