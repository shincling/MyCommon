import sys
import re
def construct_input(input_path):
    check_onetomany=False
    f_input=open(input_path,'r').readlines()
    print 'total lines of input is {}'.format(len(f_input))
    new_question_indexList=[]
    ans_indexList=[]
    old_question=''
    for idx,question in enumerate(f_input):
        spl=question.strip().split('\t')
        now_quesiton,now_ans=spl[0],spl[-1]
        if now_quesiton!=old_question:
            new_question_indexList.append(idx)
            old_question=now_quesiton
        if now_ans=='1':
            ans_indexList.append(idx)
    print 'total num of questions is :{}'.format(len(new_question_indexList))
    print 'total num of ans=1 list is {}'.format(len(ans_indexList))

    if check_onetomany and len(new_question_indexList)<len(ans_indexList):
        print 'There are some question with multi answers:(range_left,range_right)'
        for idx,que in enumerate(new_question_indexList):
            try:
                ttt=0
                ran_left=new_question_indexList[idx]
                ran_right=new_question_indexList[idx+1]
                for jj in ans_indexList:
                    if jj>ran_right:
                        break
                    if  ran_left<=jj<ran_right:
                        ttt+=1
                if ttt>1:
                    print ran_left,ran_right
            except IndexError:
                continue
    return new_question_indexList,ans_indexList

def main(input_path,result_path):
    if not (input_path and result_path):
        print 'Error in path : calculate_result.py %input_path %result_path'
    print'The baseline result of {}:\n'.format(input_path)
    construct_input(input_path)


if __name__=='__main__':
    # input_path=sys.argv[1] if sys.argv[1] else '/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/nlpcc-iccpol-2016.dbqa.training-data'
    # result_path=sys.argv[2] if sys.argv[2] else None
    input_path='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/nlpcc-iccpol-2016.dbqa.training-data'
    result_path=None
    main(input_path,result_path)