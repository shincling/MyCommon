#coding=utf8
import shelve
import numpy as np
cc=np.random.randn(200,300,300)
# bb=cc[range(64)]
# del cc
db=shelve.open('shedb') #打开直接就是dict，而且自动创建新脚本，自动读取已有脚本
for idx in range(len(cc)):
    db[str(idx)]=cc[idx] #注意这个key好像只能用str
db.close()
