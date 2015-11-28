#coding=utf8
import sys
print sys.path[0] #上一层路径

'''
__file__：当前文件路径
os.path.dirname(file): 某个文件所在的目录路径
os.path.join(a, b, c,....): 路径构造 a/b/c
os.path.abspath(path): 将path从相对路径转成绝对路径
os.pardir: Linux下相当于"../"
'''