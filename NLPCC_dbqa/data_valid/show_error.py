#coding=utf8
import sys
sys.path.append('..')
def main(train_file,test_file,train_result,test_result):
    test_file=open(test_file).readlines()
    test_result=open(test_result).readlines()
    assert len(test_file)==len(test_result)

    output_file=open('error_count_1.txt','a')
    for line,score in zip(test_file,test_result):
        spl=line.split('\t')
        question=spl[0]
        answer=spl[1]
        target=float(spl[-1].strip())
        score=float(score)
        if abs(target-score)>0.5:
            output_file.write(line.strip()+'\t'+str(score)+'\n')



    return

if __name__=='__main__':
    train_file='train7_1'
    test_file='valid3_1'
    train_result='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/results/result_0629_allmix_train'
    test_result='/home/shin/MyGit/Common/MyCommon/NLPCC_dbqa/results/result_0629_allmix_valid'
    main(train_file,test_file,train_result,test_result)

