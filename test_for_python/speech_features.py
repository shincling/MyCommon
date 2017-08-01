from python_speech_features import fbank
from python_speech_features import logfbank
import scipy.io.wavfile as wav

path='/home/sw/Shin/Codes/DL4SS_Keras/Data_with_dev/male_test.wav'
(rate,sig)=wav.read(path)
print (rate,sig)
print sig.shape
fbank_feat=fbank(sig,rate,nfilt=40)
print fbank_feat[0].shape

