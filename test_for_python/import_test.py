import time
while True:
    time.sleep(2)
    cc=open('import_test_label').read()
    if cc=='1':
        print cc
    else:
        print 'nothing'

