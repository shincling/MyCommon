def f():
    yield 'first yield msg'
    print "generator inner receive:", msg
    yield 'second yield msg'
    print "generator inner2 receive:", msg
    yield

g = f()
msg = g.next()
print "generator outer receive:", msg
# msg = g.send('first send msg')
# print "generator outer receive:", msg
msg='111'
g.send('second send msg')
g.next()
pass