#coding=utf8
import jieba.posseg as pseg
words =pseg.cut("iphone6分辨率是多少？")
words =pseg.cut("青海高原牦牛毛色有什么特点？")
words =pseg.cut("是谁把你带到孙小虎身边。")
words =pseg.cut("邓小平的妻子是谁？	")
words =pseg.cut("'潘玉利什么时候担任过任江苏省宿迁市人大常委会党组成员、秘书长、办公室主任？'")
words =pseg.cut("布什什么时候来过南京赛珍珠故居？")
words =pseg.cut("金门县议会的议员任期是多久？")
for w in words:
   print w.word,w.flag

print len('卧槽'.decode('utf8'))