#coding=utf8
import soundfile as sf
import librosa
import resampy
import numpy as np

mix_len=0
MAX_LEN=20000
spk_speech_path='0.wav'
FRAME_RATE=8000
FRAME_LENGTH = int(0.032 * FRAME_RATE)
FRAME_SHIFT = int(0.016 * FRAME_RATE)
WINDOWS= FRAME_LENGTH
signal, rate = sf.read(spk_speech_path)  # signal 是采样值，rate 是采样频率
if len(signal.shape) > 1:
    signal = signal[:, 0]
if rate != FRAME_RATE:
    # 如果频率不是设定的频率则需要进行转换
    signal = resampy.resample(signal, rate, FRAME_RATE, filter='kaiser_best')
if signal.shape[0] > MAX_LEN:  # 根据最大长度裁剪
    signal = signal[:MAX_LEN]
# 更新混叠语音长度
if signal.shape[0] > mix_len:
    mix_len = signal.shape[0]

signal -= np.mean(signal)  # 语音信号预处理，先减去均值
signal /= np.max(np.abs(signal))  # 波形幅值预处理，幅值归一化

if signal.shape[0] < MAX_LEN:  # 根据最大长度用 0 补齐,
    signal=np.append(signal,np.zeros(MAX_LEN - signal.shape[0]))

spec = np.transpose(librosa.core.spectrum.stft(signal, FRAME_LENGTH,FRAME_SHIFT))
print spec
y_signal=librosa.core.spectrum.istft(np.transpose(spec), FRAME_SHIFT,
                                        )
_mix_spec = spec
phase_mix = np.angle(_mix_spec)
_pred_spec = np.abs(_mix_spec) * np.exp(1j * phase_mix)
_pred_wav = librosa.core.spectrum.istft(np.transpose(_pred_spec), FRAME_SHIFT,
                                        window=WINDOWS)
sf.write('y_stft.wav',y_signal,FRAME_RATE) #这个是可以的了！！！！　没有istft里的window选项
sf.write('y_stft_abs_angel.wav',_pred_wav,FRAME_RATE)
