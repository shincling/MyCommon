import numpy
'''
cc=numpy.arange(10)
bb=cc.reshape(5,2)
print cc[::-1]
print bb
print bb**2
print bb==1

print numpy.concatenate((bb,bb*2),axis=1)
print numpy.hstack((bb,bb*3))
'''

c=numpy.arange(27).reshape(3,3,3)
print c
print numpy.vsplit(c,3)
print c[0,:,0]
