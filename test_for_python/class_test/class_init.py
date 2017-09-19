class ss(object):
    def __init__(self):
        self.a=1
        self.b=2
    def __pp__(self,xx):
        print 'a,b:',self.a,self.b
    def __add__(self, other):
        print 'test add'
        pass

c=ss()
c.__pp__(2)
c+c
