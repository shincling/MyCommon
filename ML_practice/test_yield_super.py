def f():
    msg = yield 'first yield msg'
    print "generator inner receive:", msg
    msg = yield 'second yield msg'
    print "generator inner receive:", msg

g = f()
msg = g.next()
print "generator outer receive:", msg
msg = g.send('first send msg')
print "generator outer receive:", msg
g.send('second send msg')