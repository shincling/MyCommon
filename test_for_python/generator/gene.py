#coding=utf8
import sys
import numpy as np

'''在外部命令行运行，1是生成器模式，0是list模式
跟我想的一样啊，根本没有节省内存啊，外面用htop可以看到两种memory占用一样的。
因为生成器挂起的时候，原来的原始数据内容也还是在memory里面，所以基本没有太大区别,
解决办法还是得想硬盘交互的，DataLoad的时候只是用来索引硬盘里的数据（最好是处理过后的数据）
目前看来生成器的适用场景在于循环内部的原始数据本身并不大，比如Fabnacci数列，只有俩变量，不断循环就得到'''


if 1:
    def gg():
        cc=np.random.randn(300,8,200,1)
        for i in cc:
            yield i
    #
    print gg
    print type(gg)
    # print '\n'
    g=gg()
    print g
    print type(g)
    print '\n',sys.getsizeof(g)
    for i in g:
        print g.next()
        aa=raw_input('press anything')
        print aa
else:
    for i in np.random.randn(300,8,200,1):
        print i
    aa=raw_input('press anything')
    print aa

