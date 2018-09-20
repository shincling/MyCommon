#coding=utf8
import numpy as np
import os
import soundfile as sf
from separation import bss_eval_sources
'''
目标文件夹path里的文件格式如下：
Notes: xxx 可以是任意字符(不用保持一致），一般用来指示说话人或者样本的标签
以下情况以2个人混合的语音为例，3个人或多个人的以此类推，需要设定 aim_mix_number 参数来更改

第1段混合语音：
0_xxx_real.wav   # 混合语音中的第一路说话人的纯净语音 (顺序无所谓) 
0_xxx_real.wav   # 混合语音中的第二路说话人的纯净语音 (顺序无所谓)
0_xxx_pre.wav    # 混合语音中预测的某一路语音(顺序无所谓)
0_xxx_pre.wav    # 混合语音中预测的某一路语音(顺序无所谓)

第2段混合语音：
1_xxx_real.wav    
1_xxx_real.wav    
1_xxx_pre.wav
1_xxx_pre.wav

第3段混合语音：
2_xxx_real.wav    
2_xxx_real.wav    
2_xxx_pre.wav
2_xxx_pre.wav

......

第k段混合语音：
k_xxx_real.wav
k_xxx_real.wav
k_xxx_pre.wav
k_xxx_pre.wav

'''

path='batch_output/'
aim_mix_number=2 #只筛选具有这个个数的混合语音（只计算有俩人合成的）
def cal(path,aim_mix_number):
    mix_number=len(set([l.split('_')[0] for l in os.listdir(path) if l[-3:]=='wav']))
    print 'num of mixed :',mix_number
    SDR_sum=np.array([])
    for idx in range(mix_number):
        pre_speech_channel=[]
        aim_speech_channel=[]
        mix_speech=[]
        for l in sorted(os.listdir(path)):
            if l[-3:]!='wav':
                continue
            if l.split('_')[0]==str(idx):
                if 'True_mix' in l:
                    mix_speech.append(sf.read(path+l)[0])
                if 'real' in l:
                    aim_speech_channel.append(sf.read(path+l)[0])
                if 'pre' in l:
                    pre_speech_channel.append(sf.read(path+l)[0])

        assert len(aim_speech_channel)==len(pre_speech_channel)
        aim_speech_channel=np.array(aim_speech_channel)
        pre_speech_channel=np.array(pre_speech_channel)
        # print aim_speech_channel.shape
        # print pre_speech_channel.shape

        result=bss_eval_sources(aim_speech_channel,pre_speech_channel)
        # result=bss_eval_sources(aim_speech_channel,aim_speech_channel)
        print result

        SDR_sum=np.append(SDR_sum,result[0])
    print 'SDR here:',SDR_sum.mean()
    return SDR_sum

cal(path,2)

