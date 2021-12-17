#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# ゼロ交差数を計算する関数
#

import matplotlib.pyplot as plt
import numpy as np
import librosa

# 音声波形データを受け取り，ゼロ交差数を計算する関数
def zero_cross(waveform):
	
	zc = 0

	for i in range(len(waveform) - 1):
		if(
			(waveform[i] > 0.0 and waveform[i+1] < 0.0) or
			(waveform[i] < 0.0 and waveform[i+1] > 0.0)
		):
			zc += 1
	
	return zc

# 音声波形データを受け取り，ゼロ交差数を計算する関数
# 簡潔版
def zero_cross_short(waveform):
	
	d = np.array(waveform)
	return sum([1 if x < 0.0 else 0 for x in d[1:] * d[:-1]])

# 配列 a の index 番目の要素がピーク（両隣よりも大きい）であれば True を返す
def is_peak(a, index):
	if(index == 0 or index == len(a) -1) :
		return False
	else :
		if(a[index - 1] <= a[index] and a[index] >= a[index + 1]):
			return True


# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load('my_aiueo2.wav', sr=SR)

# 自己相関が格納された，長さが len(x)*2-1 の対称な配列を得る
autocorr = np.correlate(x, x, 'full')

# 不要な前半を捨てる
autocorr = autocorr [len (autocorr ) // 2 : ]

# ピークのインデックスを抽出する
peakindices = [i for i in range (len (autocorr )) if is_peak (autocorr, i)]

# インデックス0 がピークに含まれていれば捨てる
peakindices = [i for i in peakindices if i != 0]

# 自己相関が最大となるインデックスを得る

max_peak_index = max (peakindices , key=lambda index: autocorr [index])

# インデックスに対応する周波数を得る

funfreq = 16000/max_peak_index



size_frame = 512			# 2のべき乗

# フレームサイズに合わせてハミング窓を作成
hamming_window = np.hamming(size_frame)

# シフトサイズ
size_shift = 16000 / 100	# 0.001 秒 (10 msec)

# スペクトログラムを保存するlist
spectrogram = []

# 基本周波数を保存するリスト
funfreqlist = []

# size_shift分ずらしながらsize_frame分のデータを取得
# np.arange関数はfor文で辿りたい数値のリストを返す
# 通常のrange関数と違うのは3つ目の引数で間隔を指定できるところ
# (初期位置, 終了位置, 1ステップで進める間隔)
for i in np.arange(0, len(x)-size_frame, size_shift):
	
	# 該当フレームのデータを取得
	idx = int(i)	# arangeのインデクスはfloatなのでintに変換
	x_frame = x[idx : idx+size_frame]
	
	

	# 窓掛けしたデータをFFT
	# np.fft.rfftを使用するとFFTの前半部分のみが得られる
	fft_spec = np.fft.rfft(x_frame * hamming_window)
	
	# np.fft.fft / np.fft.fft2 を用いた場合
	# 複素スペクトログラムの前半だけを取得
	#fft_spec_first = fft_spec[:int(size_frame/2)]
	# 【補足】
	# 配列（リスト）のデータ参照
	# list[:B] listの先頭からB-1番目までのデータを取得

	# 複素スペクトログラムを対数振幅スペクトログラムに
	fft_log_abs_spec = np.log(np.abs(fft_spec))
	# 計算した対数振幅スペクトログラムを配列に保存
	spectrogram.append(fft_log_abs_spec)
 
	zero_cross_number = zero_cross_short(x_frame) * SR // size_frame
    
	if(zero_cross_number < funfreq * 20):
		sublist = [funfreq] * int(size_shift)
		funfreqlist.extend(sublist)
	else:
		sublist = [0] * int(size_shift)
		funfreqlist.extend(sublist)


# 画像として保存するための設定
fig = plt.figure()

# スペクトログラムを描画
plt.xlabel('sample')					# x軸のラベルを設定
plt.ylabel('frequency [Hz]')		# y軸のラベルを設定
plt.imshow(
	np.flipud(np.array(spectrogram).T),		# 画像とみなすために，データを転地して上下反転
	extent=[0, len(x), 0, SR/2],			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
	aspect='auto',
	interpolation='nearest'
)
plt.plot(funfreqlist)

plt.ylim([0,1000])
plt.show()

