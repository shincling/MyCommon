#coding=utf8
def gen(N):
    for i in range(N):
        yield i**2

cc=gen(10)

def cc_next(cc):
    next(cc)
    return next(cc)

print cc_next(cc)
print next(cc)
print cc_next(cc)
print cc_next(cc)
print cc_next(cc)
print cc_next(cc)
'''以上代码说明，初始了一个生成器，不论你在何处调用它，它都好些会全局变化
在cc_next这个函数里调用了，在外部它同样改变了，有点类似与处理一个数组（地址传送）'''
