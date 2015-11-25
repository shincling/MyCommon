# -*- coding: utf-8 -*-
from elasticsearch import Elasticsearch
import ujson
import re
import os

es = Elasticsearch('127.0.0.1:9200')

path='/home/shin/MyGit/Common/MyCommon/exampleJson'
files=[]
for file in os.listdir(path):
    filename=path+'/'+file
    print filename
    files.append(filename)

    obj = ujson.load(open(filename, 'r'))
    if True:
        es.index('ntdb_test', 'companyInfo', obj, id=re.findall(r'\d+',filename))
        #print es
        #es.get('shin_0','430002')


        pass


    else:
        es.index('shin_0', 'shin_1', obj)


res = es.search(
index='ntdb_test',
doc_type='companyInfo',
body={
    'query': {
        'match':{
            'sourceWeb':'touziren'
                }
            }

    }
)
list = [x['_source'] for x in res['hits']['hits']]

for i in list:
    print i['name'].encode('utf8')