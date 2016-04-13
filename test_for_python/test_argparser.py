import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=int, default=222, help='Task#')
args=parser.parse_args()

print args