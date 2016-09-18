# !/bin/bash/
function sum()
{
a=$1
b=$2
echo `expr $a + $b`
echo $(expr $a + $b)
}
sum 3 4
sum $1 $2

function LoopPrint()
{
    count=0;
    while [ $count -lt $1 ];
    do
    echo $count;
    let ++count;
    sleep 1;
    done
    return 0;
}

read -p "Please input the times of print you want: " n;
LoopPrint $n;

