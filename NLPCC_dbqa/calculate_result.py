import sys
import re
def calculate_oneresult(input_path,result_path):
    f_input=open(input_path,'r').readlines()
    f_result=open(result_path,'r').readlines()
    assert len(f_input)==len(f_result)

    new_question_indexList=[]
    ans_indexList=[]
    old_question=''
    for idx,question in enumerate(f_input):
        spl=question.split('\t')
        now_quesiton,now_ans=spl[0],spl[-1]
        if now_quesiton!=old_question:
            new_question_indexList.append(idx)
        if now_quesiton==1:
            ans_indexList.append(idx)





def main(input_path,result_path):
    if input_path and result_path:
        calculate_oneresult(input_path,result_path)
    else:
        print 'Error in path : calculate_result.py %input_path %result_path'

    calculate_oneresult(input_path,)



if __name__=='__main__':
    input_path=sys.argv[1] if sys.argv[1] else None
    result_path=sys.argv[2] if sys.argv[2] else None
    main(input_path,result_path)