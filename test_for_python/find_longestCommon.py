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

print find_lcs_len('我是个'.decode('utf8'),'我是唉你，我是个'.decode('utf8'))