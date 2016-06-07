# -*- coding: utf8 -*-
import pickle

f_similarity=open('similarity_0606','r')
cont=f_similarity.read()
cont=cont.split('\r\n')
print 'similarity len:',len(cont)

f_overlap=open('train.QApair.WordOverlap.score','r')
overlap=f_overlap.read()
overlap=overlap.split('\r\n')
print 'overlap len:',len(overlap)

assert len(cont)==len(overlap)
result_list=pickle.load(open('result_list_0607'))
print 'result_list len:',len(result_list)

f_out=open('similarity_withLap_0607','w')
total_lines=''
for result,simi,over in zip(result_list,cont,overlap):
    one_line=result+' '+'1:{}'.format(simi)+' 2:{}\n'.format(over)
    total_lines=total_lines+one_line
f_out.write(total_lines)

