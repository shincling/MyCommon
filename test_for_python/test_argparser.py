import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=int, default=222, help='Task#')
args=parser.parse_args()

print args
def cc(task,f1=0,f2='ccc'):
    print task,f1,f2
bb={'task':'hhh'}
cc(**args.__dict__)

cc(**bb)

cc(bb)