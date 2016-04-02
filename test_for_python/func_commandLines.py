import sys
import begin
@begin.subcommand
def ttt(x,y,z='zzz'):
    print x
    print y
    print z

@begin.start
def run():
    pass

