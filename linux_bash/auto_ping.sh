# !/bin/bash
while true 
do
    sleep 500
    wget http://www.sina.com -a wgetlog
    ping sina -c 30
done
