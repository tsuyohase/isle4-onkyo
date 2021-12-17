#
# 計算機科学実験及演習 4「音響信号処理」

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

#音声ファイルを受け取り、ケプストラムの集合を返す関数
def ceps(wav_file):
    # 音声ファイルの読み込み
    x, _ = librosa.load(wav_file, sr=SR)
    # 高速フーリエ変換
    # np.fft.rfftを使用するとFFTの前半部分のみが得られる
    fft_spec = np.fft.rfft(x)
    
    #ケプストラム
    ceps = cepstrum(fft_spec)
    ceps = ceps[0:13]
    ceps = np.abs(ceps)
    return ceps

# libsoraでロードされた音声ファイルを受け取り、ケプストラムを返す関数
def ceps_sub(x):
    fft_spec = np.fft.rfft(x)
    #ケプストラム
    ceps = cepstrum(fft_spec)
    ceps = ceps[0:13]
    ceps = np.abs(ceps)
    return ceps
    

#ケプストラムの集合を受け取り、その正規分布における対数尤度を最大化する平均を返す関数
def likelihood_ave(ceps_set):
    # ave_set = [0] * (len(ceps_set[0]))
    # for n in range(len(ceps_set)):
    #     for d in range(len(ceps_set[n])):
    #         ave_set[d] = ave_set[d] + ceps_set[n][d]
    # ave_set_sub =list(map(lambda x : x/len(ceps_set), ave_set))
    ave_set = np.average(ceps_set, axis = 0)
    return ave_set
        
#ケプストラムの集合を受け取り、その正規分布における対数尤度を最大化する分散を返す関数
def likelihood_dist(ceps_set):
    # dist_set = [0] * (len(ceps_set[0]))
    # ave_set = likelihood_ave(ceps_set)
    # for n in range(len(ceps_set)):
    #     for d in range(len(ceps_set[n])):
    #         dist_set[d] = dist_set[d] + (ceps_set[n][d] -ave_set[d]) ** 2
    # dist_set_sub = list(map(lambda x : x/len(ceps_set), dist_set))
    dist_set = np.var(ceps_set, axis = 0)
    return dist_set

#ケプストラム,パラメータを受け取り、その尤度を計算する関数
def likelihood(ceps, ave_set, dist_set):
    pro_sub = 0
    for d in range(len(ceps)):
        pro_sub +=((ceps[d] - ave_set[d]) ** 2) / (2 * (dist_set[d])) 
    dist_set = np.sqrt(dist_set)
    probability = np.exp(-pro_sub) /  np.prod(dist_set)
    like = np.log(np.abs(probability))
    return like 
          
#学習データ

#学習データを入力すると平均、分散のパラメータを出力する関数
def study(wav_file):
    c_set = []
    for i in range(len(wav_file)):
        SR = 16000
        x, _ = librosa.load(wav_file[i], sr = SR)
        size_frame = 512
        size_shift = 16000/100
        for j in np.arange(0, len(x)-size_frame, size_shift):
            idx = int(j)	# arangeのインデクスはfloatなのでintに変換
            x_frame = x[idx : idx+size_frame]
            cep = ceps_sub(x_frame)
            c_set.append(cep) 
    ave = likelihood_ave(c_set)
    dist = likelihood_dist(c_set)

    return ave, dist



a_ave_set, a_dist_set = study(['a.wav'])
i_ave_set, i_dist_set = study(['i.wav'])
u_ave_set, u_dist_set = study(['u.wav'])
e_ave_set, e_dist_set = study(['e.wav'])
o_ave_set, o_dist_set = study(['o.wav'])


# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load('aiueo.wav', sr=SR)

size_frame = 512			# 2のべき乗

# フレームサイズに合わせてハミング窓を作成
hamming_window = np.hamming(size_frame)

# シフトサイズ
size_shift = 16000 / 100	# 0.001 秒 (10 msec)

# スペクトログラムを保存するlist
spectrogram = []

#識別結果を保存する
recoginition = []


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
	
	# 複素スペクトログラムを対数振幅スペクトログラムに
	fft_log_abs_spec = np.log(np.abs(fft_spec))
	# 計算した対数振幅スペクトログラムを配列に保存
	spectrogram.append(fft_log_abs_spec)
 
    
 
	a_likelihood = likelihood(ceps_sub(x_frame),a_ave_set,a_dist_set)
	i_likelihood = likelihood(ceps_sub(x_frame),i_ave_set,i_dist_set)
	u_likelihood = likelihood(ceps_sub(x_frame),u_ave_set,u_dist_set)
	e_likelihood = likelihood(ceps_sub(x_frame),e_ave_set,e_dist_set)
	o_likelihood = likelihood(ceps_sub(x_frame),o_ave_set,o_dist_set)
 
	list = [a_likelihood,i_likelihood,u_likelihood,e_likelihood,o_likelihood]
	max_like = max(list)
	if(a_likelihood == max_like) :
		sublist = [0] * int(size_shift)
		recoginition.extend(sublist)
	elif(i_likelihood == max_like) :
		sublist = [1000] * int(size_shift)
		recoginition.extend(sublist)
	elif(u_likelihood == max_like) :
		sublist = [2000] * int(size_shift)
		recoginition.extend(sublist)
	elif(e_likelihood == max_like) :
		sublist = [3000] * int(size_shift)
		recoginition.extend(sublist)
	elif(o_likelihood == max_like) :
		sublist = [4000] * int(size_shift)
		recoginition.extend(sublist)
       


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
plt.plot(recoginition)			# 描画

plt.show()







