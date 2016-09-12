# !/bin/bash
# This is a test script , about Hello,world.
echo 'Hello World'

# Input Function Test
echo 'Please input your name:'
read NAME
echo Welcome,$NAME
#这个和上面是一样的 加不加引号都可以
echo "Welcome,$NAME"

#传递形参 例如 test a b c 
echo auto Input is : $1 $2 $3
echo 'The position is :'$0
