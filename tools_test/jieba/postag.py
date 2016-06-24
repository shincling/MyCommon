#coding=utf8
import jieba.posseg as pseg
words =pseg.cut("iphone6分辨率是多少？")
words =pseg.cut("是谁把你带到孙小虎身边。")
words =pseg.cut("青海高原牦牛毛色有什么特点？")
words =pseg.cut("请问霍建华喜欢哪3只小狗  ")
for w in words:
   print w.word,w.flag