#coding:utf8
import sys
import re


def hhh(x,y):
    print x+y

#hhh(input('please the first:'),input('please the first:'))
expression=r'''location=北京-北京市&&做市商=xxx&&主券商=XXXX&&crew=<int>'''
print  re.findall(r'location=([^&|]*)',expression)

'''
a1=sys.argv[1]
a2=sys.argv[2]
hhh(float(a1),float(a2))



print "let's start."

cc=raw_input('hi')
print cc*3
aa=raw_input('hi2')
print 'hhh'
'''

xx=10
def addone(x):
    x=x+1

addone(xx)
pass