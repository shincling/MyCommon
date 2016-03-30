def findallindex(string,seq):
    start=0
    end=len(string)
    position=[]
    while True:
        index=string.find(seq,start,end)
        #print index
        if index!=-1:
            position.append(index)
            start=index+1
        else :
            break 
    return position

if __name__=='__main__':
    aa='ljkjljljdlkjlkjjdljkjljdjlklkjd'
    print aa
    cc=findallindex(aa,'d')
