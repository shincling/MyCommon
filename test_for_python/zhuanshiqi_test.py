import time
'''
def watcher(func):
    def timecoutn(a):
        print '%s ,started:%s'%(time.ctime(),func.__name__)
        fff=func(a)
        print 'after'
        return fff
    return timecoutn
@watcher
def foo(aaa):
    print aaa

foo('sss')
time.sleep(2)
foo('sss')
'''

def funParse(fun):
    #print fun(333)
    def ss(aa):
        tt=fun(aa)
        return tt
    print ss(22)

def ff(int):
    return 5*int

funParse(ff)