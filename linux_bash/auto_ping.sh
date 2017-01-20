# !/bin/bash
while true 
do
    sleep 500
    wget http://wwww.sina.com -a wgetlog
    ping sina -c 30
done
