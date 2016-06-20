#coding=utf8
import jieba.posseg as pseg
words =pseg.cut("iphone6分辨率是多少？")
words =pseg.cut("是谁把你带到孙小虎身边。")
for w in words:
   print w.word,w.flag