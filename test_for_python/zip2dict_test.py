a=range(10)
b=range(-10,0,1)
cc=zip(a,b)
print type(cc)
print cc

c=dict(cc)
print type(c)
print c[1]
