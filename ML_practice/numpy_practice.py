import numpy

cc=numpy.arange(10)
bb=cc.reshape(5,2)
print cc[::-1]
print bb
print bb**2
print bb==1

print numpy.concatenate((bb,bb*2),axis=1)
print numpy.hstack((bb,bb*3))
