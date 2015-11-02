import os
import re
dic=[]
path=r'D:\DeepLearning\Êý¾Ý¼¯\Facebook QA\tasks_1-20_v1-2\en'
dic=[os.path.join(path,s) for s in os.listdir(path)]

'''
f=open(dic[1])

print os.path.getsize(dic[1])
print dic[1]
conten=f.read()
print len(conten)
'''


for txt in dic:
    content=open(txt).read()
    fin=[]
    batch=[]
    index=[]
    start=0
    while True:
        ind=content.find('\n1 ',start)
        if ind!=-1:
            index.append(ind)
            start=ind+1
        else:
            break
    index.append(len(content))
    index.insert(0,-1)
    for i in range(len(index)-1):
        batch.append(content[(index[i]+1):(index[i+1]+1)])

    for i in range(len(batch)):
        c1=[]
        c2=[]
        lines=re.split('\n',batch[i])
        for line in lines:
            if line.find('\t')!=-1:
                c2.append(re.split('\t',line))
            else:
                c1.append(line)
        fin.append([c1,c2])
        
        
        
