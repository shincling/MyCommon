import jieba
import re

f=open('/home/shin/DeepLearning/MemoryNetwork/QA/planeplane.txt','r')
content=f.read().decode('gbk').encode('utf8')
fw=open('/home/shin/DeepLearning/MemoryNetwork/QA/planeplane_shin','w')
content=content.split('dialogue ')
for i in range(1,5000):
    sent_id=1
    dia=content[i][(content[i].index('\n')+1):-4]
    dia=dia.split('\r\n')
    w_dia=''
    for j in range(len(dia)):
        '''
        if j%2==0:
            sent_w=
            assert 'M' in dia[j]
        '''
        slot_count='nil'
        slot_time='nil'
        slot_idnumber='nil'
        slot_destination='nil'
        slot_departure='nil'
        slot_name='nil'

        sent=dia[j][3:-7]


        if 'greeting' in sent:

            w_sent=str(sent_id)
            sent_id=sent_id+1
            sent_list=jieba._lcut(sent[:sent.index('\t')])
            for word in (sent_list):
                w_sent +=' '
                w_sent +=word
                w_sent +='\n'
            w_dia=w_dia+w_sent
        elif 'request' in sent:
            pass
        elif 'inform' in sent:
            if 'departure' in sent:
                slot_departure=re.findall('departure=([^i\t+])',sent)[0]+'\t'+str(sent_id)
            else:
                slot_departure='nil'+'\t'+str(sent_id)
            if 'destination' in sent:
                slot_destination=re.findall('destination=([^i\t+])',sent)[0]+'\t'+str(sent_id)
            else:
                slot_destination='nil'+'\t'+str(sent_id)
            if 'time' in sent:
                slot_time=re.findall('time=([^i\t+])',sent)[0]+'\t'+str(sent_id)
            else:
                slot_time='nil'+'\t'+str(sent_id)
            if 'count' in sent:
                slot_count=re.findall('count=([^i\t+])',sent)[0]+'\t'+str(sent_id)
            else:
                slot_count='nil'+'\t'+str(sent_id)
            if 'idnumber' in sent:
                slot_idnumber=re.findall('idnumber=([^i\t+])',sent)[0]+'\t'+str(sent_id)
            else:
                slot_idnumber='nil'+'\t'+str(sent_id)
            if 'name' in sent:
                slot_name=re.findall('name=([^i\t+])',sent)[0]+'\t'+str(sent_id)
            else:
                slot_name='nil'+'\t'+str(sent_id)





            w_sent=str(sent_id)
            sent_id=sent_id+1
            sent_list=jieba._lcut(sent[:sent.index('\t')])
            for word in (sent_list):
                w_sent +=' '
                w_sent +=word
                w_sent +='\n'
            w_dia=w_dia+w_sent

        if j%2==1:
            w_dia=str(sent_id)+w_dia+'departure ?\t%s\n'%slot_departure
            sent_id=sent_id+1
            w_dia=str(sent_id)+w_dia+'destination ?\t%s\n'%slot_destination
            sent_id=sent_id+1
            w_dia=str(sent_id)+w_dia+'name ?\t%s\n'%slot_name
            sent_id=sent_id+1
            w_dia=str(sent_id)+w_dia+'idnumber ?\t%s\n'%slot_idnumber
            sent_id=sent_id+1
            w_dia=str(sent_id)+w_dia+'time ?\t%s\n'%slot_time
            sent_id=sent_id+1
            w_dia=str(sent_id)+w_dia+'count ?\t%s\n'%slot_count
            sent_id=sent_id+1

















    pass
