#coding=utf8
import jieba
import jieba.analyse
sentence='我爱北京天安门，天安门上太阳升。'
topK=20
print '\n'.join(jieba.analyse.extract_tags(sentence,topK))

