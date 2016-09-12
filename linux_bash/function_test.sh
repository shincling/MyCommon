#!/bin/bash
hello()
{
echo Hello,world
}

function hi()
{
echo Hi,world
}

hello
hi

function plus()
{
#echo 'expr $1 + $2'
E= expr $1 + $2
}
#D= 'plus 100 200'
#echo 'D='$D

# 这里传递参数有各种问题 之后要专门找个地方好好看一看，加引号完全好像没办法work的
plus 100 300
echo 'E='$E
