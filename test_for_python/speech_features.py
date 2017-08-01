#coding=utf8
from python_speech_features import fbank
from python_speech_features import logfbank
import scipy.io.wavfile as wav

path='/home/sw/Shin/Codes/DL4SS_Keras/Data_with_dev/male_test.wav'
(rate,sig)=wav.read(path)
print (rate,sig)
print sig.shape #43520
fbank_feat=fbank(sig,rate,winstep=0.01,nfilt=40)
print fbank_feat[0].shape  # 271的结果是这样得到噢的，43520/(0.01s*16000)

