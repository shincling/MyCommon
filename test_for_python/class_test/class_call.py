class ss(object):
    def __init__(self):
        self.a=1
        self.b=2

    def __call__(self,x=''):
        print 'call this object.',x


a=ss()
a(12)
a()
