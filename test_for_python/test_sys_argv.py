import sys

def run():
    print 'ddd'
    print(''.join(sys.argv))

if __name__=="__main__":
    print 'thhtt'
    cc=sys.argv
    run()
    for idx,i in enumerate(cc):
        print idx
        print i
        print '\n'

