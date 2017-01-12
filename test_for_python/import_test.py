import time
# while True:
for i in range(1000000):
    time.sleep(2)
    cc=int(open('import_test_label').read())
    # print cc
    pos=i%cc
    if pos==0:
        print i
    else:
        print 'nothing'

