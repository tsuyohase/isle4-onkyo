#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# ケプストラムを計算する関数
#

import numpy as np

import matplotlib.pyplot as plt
import librosa


# スペクトルを受け取り，ケプストラムを返す関数
def cepstrum(amplitude_spectrum):
	log_spectrum = np.log(np.abs(amplitude_spectrum))
	cepstrum = np.fft.fft(log_spectrum)
	return cepstrum


# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load('o.wav', sr=SR)

# 高速フーリエ変換
# np.fft.rfftを使用するとFFTの前半部分のみが得られる
fft_spec = np.fft.rfft(x)

# 複素スペクトログラムを対数振幅スペクトログラムに
fft_log_abs_spec = np.log(np.abs(fft_spec))

ceps = cepstrum(fft_spec)

ceps[14:len(ceps)-1] = 0 

ceps_spec= np.fft.ifft(ceps)  

fig = plt.figure()

plt.xlabel('frequency [Hz]')		# x軸のラベルを設定
plt.ylabel('amplitude')				# y軸のラベルを設定
plt.xlim([0, SR/2])					# x軸の範囲を設定

x_data = np.linspace((SR/2)/len(fft_log_abs_spec), SR/2, len(fft_log_abs_spec))
		# 描画
plt.plot(x_data,fft_log_abs_spec)
plt.plot(x_data, ceps_spec)	

# 表示
plt.show()

fig.savefig('plot-cepstrum-o.png')



