#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 音声ファイルを読み込み，フーリエ変換を行う．
#

# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa
import math


# 周波数からノートナンバーへ変換（notenumber.pyより）
def hz2nn(frequency):
    	return int (round (12.0 * (math.log(frequency / 440.0) / math.log (2.0)))) + 69

# def shs(spectrum, frequencies, candidate):
#     sh = np.zeros(len(candidate))
#     for s, f in zip (spectrum, frequencies):
#         nn = hz2nn(f)
#         for c in range (len(candidate)):
#             for nc in np.arange(candidate[c], 128, 12):
#                 if (nn == int(nc)):
#                     sh[c] += np.abs(s)
#     return sh
 
def chroma_vector(spectrum, frequencies):
    	
	# 0 = C, 1 = C#, 2 = D, ..., 11 = B

	# 12次元のクロマベクトルを作成（ゼロベクトルで初期化）
	cv = np.zeros(128)
	
	# スペクトルの周波数ビン毎に
	# クロマベクトルの対応する要素に振幅スペクトルを足しこむ
	for s, f in zip (spectrum , frequencies):
		nn = hz2nn(f)
		cv[nn] += np.abs(s)
	
	return cv       

def nn2hz(nn):
    return int(440 * (2 ** ((nn - 69)/12)))

def shs(cv,candidate):
    sh = np.zeros(len(candidate))
    for c in range(len(candidate)):
        for nc in np.arange(candidate[c], 128, 12):
            sh[c] += cv[nc]
    return sh
# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load('shs-test-man.wav', sr=SR)

# 高速フーリエ変換
# np.fft.rfftを使用するとFFTの前半部分のみが得られる
spec = np.fft.rfft(x)

# 複素スペクトログラムを対数振幅スペクトログラムに
log_abs_spec = np.log(np.abs(spec))


size_frame = 512			# 2のべき乗

# フレームサイズに合わせてハミング窓を作成
hamming_window = np.hamming(size_frame)

# シフトサイズ
size_shift = 16000 / 100	# 0.001 秒 (10 msec)

# スペクトログラムを保存するlist
spectrogram = []

candidate = range(36,60)
shs_list = []



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
 
	# s = shs(fft_spec,frequencies,candidate)
	s = chroma_vector(fft_spec,frequencies)
	sh = shs(s,candidate) 
	
	max_index = np.argmax(sh) +candidate[0]
	fre = nn2hz(max_index)
	sublist = [fre] * int(size_shift)
	shs_list.extend(sublist)
    


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

plt.plot(shs_list)
plt.ylim([0,1000])
plt.show()

#
# スペクトルを画像に表示・保存
#

# 画像として保存するための設定
fig = plt.figure()

# スペクトログラムを描画
plt.xlabel('frequency [Hz]')		# x軸のラベルを設定
plt.ylabel('amplitude')				# y軸のラベルを設定
plt.xlim([0, SR/2])					# x軸の範囲を設定
# x軸のデータを生成（元々のデータが0~8000Hzに対応するようにする）
x_data = np.linspace((SR/2)/len(log_abs_spec), SR/2, len(log_abs_spec))
plt.plot(x_data, log_abs_spec)			# 描画
# 【補足】
# 縦軸の最大値はサンプリング周波数の半分 = 16000 / 2 = 8000 Hz となる

# 表示
plt.show()

# 画像ファイルに保存
# fig.savefig('plot-spectrum-whole.png')

# # 横軸を0~2000Hzに拡大
# # xlimで表示の領域を変えるだけ
# fig = plt.figure()
# plt.xlabel('frequency [Hz]')
# plt.ylabel('amplitude')
# plt.xlim([0, 2000])
# plt.plot(x_data, fft_log_abs_spec)

# # 表示
# plt.show()

