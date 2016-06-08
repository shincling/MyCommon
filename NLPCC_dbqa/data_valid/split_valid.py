#coding=utf8
import sys
sys.path.append('..')
import calculate_result

input_path='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/nlpcc-iccpol-2016.dbqa.training-data'
result_path=None
question_list,ans_list,f_input=calculate_result.main(input_path,result_path)
print len(question_list)
print len(ans_list)
full_list=question_list+[len(f_input)]
question_rangelist=[]
for idx,ques in enumerate(full_list):
    try:
        range=(ques,full_list[idx+1])
    except IndexError:
        break
    question_rangelist.append(range)

