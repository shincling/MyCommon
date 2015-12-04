import time

def watcher(func):
    print ('before the func')
    def timecoutn(a):
        print '%s ,started:%s'%(time.ctime(),func.__name__)
        fff=func(a)
        return fff
    return timecoutn
@watcher
def foo(aaa):
    print aaa


foo('sss')
time.sleep(2)


foo('sss')