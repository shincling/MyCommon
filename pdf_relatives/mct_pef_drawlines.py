# -*- coding: utf8 -*-
__author__ = 'shin'

import re
import reportlab
import pyPdf
import reportlab_test
#canvas画图的类库
from reportlab.pdfgen.canvas import Canvas

def find_horizon(real_linescode):
    dict_horizon={}
    for i, line in enumerate(real_linescode):
        if abs(line[1]-line[3])<1:
            dict_horizon[str(i)]=(line[0],round((line[1]+line[3])/2,3),line[2],round((line[1]+line[3])/2,3))
    return dict_horizon

def find_vertical(real_linescode):
    dict_vertical={}
    for i, line in enumerate(real_linescode):
        if abs(line[0]-line[2])<1:
            dict_vertical[str(i)]=(round((line[0]+line[2])/2,3),line[1],round((line[0]+line[2])/2,3),line[3])
    return dict_vertical

def draw(list):
    #声明Canvas类对象，传入的就是要生成的pdf文件名字
    for plot in list:


        can = Canvas('report.pdf')
        can.line(plot[0],plot[1],plot[2],plot[1])#最上面的横线
        can.line(plot[0],plot[3],plot[2],plot[3])#最下面的横线
        can.line(plot[0],plot[1],plot[0],plot[3])#最左面的横线
        can.line(plot[2],plot[1],plot[2],plot[3])#最右面的横线



        can.save()


xmlreader = open('/home/shin/Memect/ntdb/page50.xml', 'r').read()
linescode=re.findall('''<rect linewidth="0" bbox="([^,]*),([^,]*),([^,]*),([^"]*)"''',xmlreader)
real_linescode=[]
for line in linescode:
    linecode=(float(line[0]),float(line[1]),float(line[2]),float(line[3]))
    real_linescode.append(linecode)
dict_horizon=find_horizon(real_linescode)
dict_vertical=find_vertical(real_linescode)

'''删除只是一个点的元素'''
for keys in dict_horizon.keys():
    if keys in dict_vertical:
        print 'The lines with just a point:%s--[%s]'%(keys,str(real_linescode[int(keys)]))
        dict_horizon.pop(keys)
        dict_vertical.pop(keys)


vvv=[]
for keys in dict_vertical:
    vvv.append([dict_vertical[keys][1],dict_vertical[keys][3]])
vvv.sort()

ran_ver={} #用来储存表的范围的字典，key是序号，value是纵向范围值
current_range=0
ran_ver[str(current_range)]=vvv[0]
for vv in vvv:
    if vv[0]<=ran_ver[str(current_range)][1]+1:
        ran_ver[str(current_range)]=[ran_ver[str(current_range)][0],max(vv[1],ran_ver[str(current_range)][1])]
    else :
        current_range+=1
        ran_ver[str(current_range)]=vv

total_table=current_range+1
print '\nThe total number of table : %d'%(total_table)

ran_hon={}
for i in range(total_table):
    for j,hhh in enumerate(dict_horizon):
        if j==0:
            left=dict_horizon[hhh][0]
            right=dict_horizon[hhh][2]
        if ran_ver[str(i)][0]<=dict_horizon[hhh][1]<=ran_ver[str(i)][1]:
            if dict_horizon[hhh][0]<left:
                left=dict_horizon[hhh][0]
            if dict_horizon[hhh][2]>right:
                right=dict_horizon[hhh][2]
    ran_hon[str(i)]=[left,right]

ran=[]
for i in range(total_table):
    ran.append((ran_hon[str(i)][0],ran_ver[str(i)][0],ran_hon[str(i)][1],ran_ver[str(i)][1]))

draw(ran)

#reportlab_test.pdf_head()
pass


