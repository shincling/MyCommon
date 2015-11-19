'''# -*- coding:utf8 -*-'''
from pyquery import PyQuery as pq
import lxml

page=pq(url='http://yunvs.com/list/mai_1.html')

tr=page('tr')

td=page('td')
print len(td)

for data in tr:
    #print pq(data).text()
    for i in range(len(data)):
        print pq(data).find('td').eq(i).text()


