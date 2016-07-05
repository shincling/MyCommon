#coding=utf8
def find_lcs_len(s1, s2):
    m = [[ 0 for x in s2] for y in s1]
    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:
                if p1 == 0 or p2 == 0:
                    m[p1][p2] = 1
                else:
                    m[p1][p2] = m[p1-1][p2-1]+1
            elif m[p1-1][p2] < m[p1][p2-1]:
                m[p1][p2] = m[p1][p2-1]
            else:               # m[p1][p2-1] < m[p1-1][p2]
                m[p1][p2] = m[p1-1][p2]
    return m[-1][-1]

def find_lcs(s1, s2):# 长度表:每个元素被设置为零。
    m = [ [ 0 for x in s2 ] for y in s1 ] # 方向表： 1st bit for p1, 2nd bit for p2.
    d = [ [ None for x in s2 ] for y in s1 ]
    # a negative index always gives an intact zero.
    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:
                if p1 == 0 or p2 == 0:
                    m[p1][p2] = 1
                else:
                    m[p1][p2] = m[p1-1][p2-1]+1
                d[p1][p2] = 3
            elif m[p1-1][p2] < m[p1][p2-1]:
                m[p1][p2] = m[p1][p2-1]
                d[p1][p2] = 2
            else:
                m[p1][p2] = m[p1-1][p2]
                d[p1][p2] = 1
    (p1, p2) = (len(s1)-1, len(s2)-1)
# now we traverse the table in reverse order.
#<span style="font-family: Arial, Verdana, sans-serif;">www.iplaypython.com</span>
    s = []
    while 1:
        # print p1,p2
        c = d[p1][p2]
        if c == 3: s.append(s1[p1])
        if not ((p1 or p2) and m[p1][p2]): break
        if c & 2: p2 -= 1
        if c & 1: p1 -= 1
    s.reverse()
    for i in s:
        print i
    return ''.join(s)

# print find_lcs('我是,sdfsdf,我是唉你'.decode('utf8'),'我sdfsdf是唉你，我是个'.decode('utf8'))
print find_lcs('我是唉你sdfsdf'.decode('utf8'),'我sdfsdf是唉你，我是个'.decode('utf8'))
