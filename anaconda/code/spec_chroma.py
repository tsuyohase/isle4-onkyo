
# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa
import math


def hz2nn(frequency):
    	return int (round (12.0 * (math.log(frequency / 440.0) / math.log (2.0)))) + 69

def chroma_vector(spectrum, frequencies):
    	
	# 0 = C, 1 = C#, 2 = D, ..., 11 = B

	# 12次元のクロマベクトルを作成（ゼロベクトルで初期化）
	cv = np.zeros(12)
	
	# スペクトルの周波数ビン毎に
	# クロマベクトルの対応する要素に振幅スペクトルを足しこむ
	for s, f in zip (spectrum , frequencies):
		nn = hz2nn(f)
		cv[nn % 12] += np.abs(s)
	
	return cv

def lc(cv,a):
    l = np.zeros(24)
    for c in range(len(cv)):
        l[c] = a[0] * cv[c % 12] + a[1] * cv[(c + 4) % 12 ] + a[2] *cv[(c + 7)% 12]
        l[c + 12] = a[0] * cv[c % 12] + a[1] * cv[(c +3) % 12] + a[2] * cv[(c + 7) % 12]
    return l

def nn2hz(nn):
    return int(440 * (2 ** ((nn - 69)/12)))
# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load('easy_chords.wav', sr=SR)

#
# 短時間フーリエ変換
#

# フレームサイズ
size_frame = 512			# 2のべき乗

# フレームサイズに合わせてハミング窓を作成
hamming_window = np.hamming(size_frame)

# シフトサイズ
size_shift = 16000 / 100	# 0.001 秒 (10 msec)

# スペクトログラムを保存するlist
spectrogram = []

lc_list = []



# size_shift分ずらしながらsize_frame分のデータを取得
# np.arange関数はfor文で辿りたい数値のリストを返す
# 通常のrange関数と違うのは3つ目の引数で間隔を指定できるところ
# (初期位置, 終了位置, 1ステップで進める間隔)
for i in np.arange(0, len(x)-size_frame, size_shift):
	
	# 該当フレームのデータを取得
	idx = int(i)	# arangeのインデクスはfloatなのでintに変換
	x_frame = x[idx : idx+size_frame]
	
	# 【補足】
	# 配列（リスト）のデータ参照
	# list[A:B] listのA番目からB-1番目までのデータを取得

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
 
	frequencies = np.linspace(8000/len(fft_spec), 8000, len(fft_spec))
	cv = chroma_vector(fft_spec, frequencies)
	a = [1,0.5,0.8]  
	l = lc(cv, a)
	max_index = np.argmax(l)
	sublist = [max_index] * int(size_shift)
	lc_list.extend(sublist)
	
 
 
 
 


#
# スペクトログラムを画像に表示・保存
#

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
# plt.ylim([0,1000])
plt.show()

fig = plt.figure()
plt.xlabel('sample')					# x軸のラベルを設定
plt.ylabel('Lc')
plt.plot(lc_list)

plt.show()


# 【補足】
# 縦軸の最大値はサンプリング周波数の半分 = 16000 / 2 = 8000 Hz となる

# 画像ファイルに保存
# fig.savefig('plot-spectogram_1000_aiueo2.png')


