from elasticsearch import Elasticsearch
import ujson
import re
import sys

es = Elasticsearch('127.0.0.1:9200')

filename='/home/shin/MyGit/Common/MyCommon/430002.json'
obj = ujson.load(open(filename, 'r'))
print filename
if True:
    es.index('investor', 'investor', obj, id=re.findall(r'\d+',filename))
    print es
else:
    es.index('investor', 'investor', obj)

