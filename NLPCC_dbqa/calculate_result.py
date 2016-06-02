import sys
import re
def construct_input(input_path):
    f_input=open(input_path,'r').readlines()
    new_question_indexList=[]
    ans_indexList=[]
    old_question=''
    for idx,question in enumerate(f_input):
        spl=question.strip().split('\t')
        now_quesiton,now_ans=spl[0],spl[-1]
        if now_quesiton!=old_question:
            new_question_indexList.append(idx)
        if now_quesiton=='1':
            ans_indexList.append(idx)
    print len(new_question_indexList),len(ans_indexList)
    return new_question_indexList,ans_indexList




def main(input_path,result_path):
    if not (input_path and result_path):
        print 'Error in path : calculate_result.py %input_path %result_path'
    print'The baseline result of {}:\n'.format(input_path)
    quesion_list,ans_list=construct_input(input_path)


if __name__=='__main__':
    # input_path=sys.argv[1] if sys.argv[1] else '/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/nlpcc-iccpol-2016.dbqa.training-data'
    # result_path=sys.argv[2] if sys.argv[2] else None
    input_path='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/nlpcc-iccpol-2016.dbqa.training-data'
    result_path=None
    main(input_path,result_path)