#coding=utf8
import sys
sys.path.append('..')
import random
import calculate_result

ratio=0.5
shuffle_times=10

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

content_list=[]
for range in question_rangelist:
    content_list.append(f_input[range[0]:range[1]])
print 'total content covers {} stories.'.format(len(content_list))

# for _ in range(10):
#     random.shuffle(content_list)
#     print 'Shuffle ~~~~~  ! '
#
random.shuffle(content_list)
print 'Shuffle ~~~~~  ! '
random.shuffle(content_list)
print 'Shuffle ~~~~~  ! '
random.shuffle(content_list)
print 'Shuffle ~~~~~  ! '
random.shuffle(content_list)
print 'Shuffle ~~~~~  ! '
random.shuffle(content_list)
print 'Shuffle ~~~~~  ! '

part_index=int(len(content_list)*(1-ratio))
print 'The train coves {} stories,and the valid covers {} stories'.format(part_index,len(content_list)-part_index)

f_train=open('training5_1','w')
f_valid=open('valid5_1','w')

for idx,one_story in enumerate(content_list):
    if idx<=part_index:
        for line in one_story:
            f_train.write(line)
    else:
        for line in one_story:
            f_valid.write(line)


