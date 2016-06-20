#coding=utf8
import jieba.posseg as pseg
words =pseg.cut("iphone6分辨率是多少？")
for w in words:
   print w.word,w.flag