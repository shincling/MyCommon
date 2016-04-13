import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=int, default=222, help='Task#')
args=parser.parse_args()

print args
def cc(task,f1=0,f2='ccc'):
    print task,f1,f2

def cc1(f1,f2,task='ccc'):
    print f1,f2,task
bb={'task':'hhh'}
bb1=(11,22,33)
cc(**args.__dict__)

cc(**bb)

cc(bb)

cc(*bb1)

bb2=(22,33)

cc1(*bb2,**bb)