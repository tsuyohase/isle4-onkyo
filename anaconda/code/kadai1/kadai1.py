import librosa
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import tkinter
import threading
import time
from pydub import AudioSegment
from pydub.utils import make_chunks


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ピークを求める関数
def is_peak(a, index):
    if(index == 0 or index == len(a) - 1) :
        return False
    else :
        return a[index - 1] < a[index] and a[index] > a[index + 1]

# ゼロ交差数を求める関数
def zero_cross_short(waveform):
    d = np.array(waveform)
    return sum([1 if x < 0.0 else 0 for x in d[1:] * d[:-1]])

# ケプストラム
def cepstrum(amplitude_spectrum):
    log_spectrum = np.log(np.abs(amplitude_spectrum))
    return np.fft.fft(log_spectrum)
    
# 母音識別のための学習関数。音声ファイルを受け取り、推定されたパラメータを返す
def study(wav_file):
    
    # 短時間フレームごとの音声データのケプストラムを格納する配列
    study_ceps_set = [] 
    for i in range(len(wav_file)):
        study_x, _ = librosa.load(wav_file[i], sr = SR)
        for j in np.arange(0, len(study_x) - size_frame, size_shift):
            
            # フレームごとの音声データを切り出す
            jdx = int(j)
            study_x_frame = study_x[jdx : jdx + size_frame]
            
            # それぞれのフレームのケプストラムを計算し、配列に格納する
            study_ceps =  cepstrum(np.fft.rfft(study_x_frame * hamming_window))
            study_ceps = np.abs(study_ceps[0:13])
            study_ceps_set.append(study_ceps)
    # フレームごとのケプストラムから最尤推定し、平均、分散のパラメータを求める
    like_ave = np.average(study_ceps_set, axis = 0)
    like_dist = np.var(study_ceps_set, axis = 0)
    return like_ave, like_dist
    
# 尤度を求める関数
def likelihood(x_frame_data, ave_set, dist_set):
    
    # 受け取ったデータからケプストラムを求める
    ceps_data = cepstrum(np.fft.rfft(x_frame_data * hamming_window))
    ceps_data = np.abs(ceps_data[0:13])
    
    # 尤度を求める
    like_sub = 0 # 式が複雑なので小分けにして計算する
    for d in range(len(ceps_data)):
        like_sub += ((ceps_data[d] - ave_set[d]) ** 2) / (2 * (dist_set[d]))
    dist_set = np.sqrt(dist_set)
    like = np.exp(-like_sub) / np.prod(dist_set)
    return np.log(np.abs(like))
    
size_frame = 4096 # フレームサイズ
SR = 16000 # サンプリングレート
size_shift = 16000 / 100 # シフトサイズ

# 音声ファイルを読み込む
x, _ = librosa.load('aiueo.wav', sr =SR)

# 音声ファイルの再生時間を求める
duration = len(x) / SR

# ハミング窓
hamming_window = np.hamming(size_frame)

# スペクトログラム格納用
spectrogram = []

# 音量格納用
volume = []

# 基本周波数格納用
funfreq = []

# 学習結果
a_ave_set, a_dist_set = study(['a.wav'])
i_ave_set, i_dist_set = study(['i.wav'])
u_ave_set, u_dist_set = study(['u.wav'])
e_ave_set, e_dist_set = study(['e.wav'])
o_ave_set, o_dist_set = study(['o.wav'])

# 識別格納用
recognition = []

# フレームごとにスペクトログラム、音量、基本周波数、ゼロ交差数による識別、母音識別を行う
for i in np.arange(0, len(x) - size_frame, size_shift):
    
    # フレームごとに音声ファイルを切り出す
    idx = int(i)
    x_frame = x[idx : idx + size_frame]    
    
    # スペクトログラム
    fft_spec = np.fft.rfft(x_frame * hamming_window)
    fft_log_abs_spec = np.log(np.abs(fft_spec))
    spectrogram.append(fft_log_abs_spec)
    
    # 音量
    vol = 20 * np.log10(np.mean(x_frame ** 2))
    volume.append(vol)
    
    # 基本周波数(自己相関による推定)
    autocorr = np.correlate(x_frame, x_frame, 'full') # 自己相関を求める
    autocorr = autocorr [len(autocorr) // 2 :] # 前半は捨てる
    peakindices = [i for i in range (len (autocorr)) if is_peak (autocorr, i)] #ピークを求める
    peakindices = [i for i in peakindices if i != 0] # 0が含まれていたら捨てる
    max_peak_index = max(peakindices , key=lambda index: autocorr [index]) # 最大のピークを求める
    ff = SR/max_peak_index # 逆数が基本周波数
    
    # ゼロ交差数
    zero_cross_number = zero_cross_short(x_frame) * SR // size_frame
    # ゼロ交差数による推定を行い、有声音の部分には基本周波数、無声音の部分には0を格納する
    if(ff * 3 < zero_cross_number):
        funfreq.append(ff)
    else :
        funfreq.append(0)
    
    # 母音識別
    # 尤度を求める
    a_likelihood = likelihood(x_frame, a_ave_set, a_dist_set)
    i_likelihood = likelihood(x_frame, i_ave_set, i_dist_set)
    u_likelihood = likelihood(x_frame, u_ave_set, u_dist_set)
    e_likelihood = likelihood(x_frame, e_ave_set, e_dist_set)
    o_likelihood = likelihood(x_frame, o_ave_set, o_dist_set)
    # それぞれの母音の尤度の最大値を求め、それぞれの母音に対応させた数字を格納する
    like = [a_likelihood,i_likelihood,u_likelihood,e_likelihood,o_likelihood]
    max_like = max(like)
    if(a_likelihood == max_like):
        recognition.append(0)
    elif(i_likelihood == max_like):
        recognition.append(1)
    elif(u_likelihood == max_like):
        recognition.append(2)
    elif(e_likelihood == max_like):
        recognition.append(3)
    elif(o_likelihood == max_like):
        recognition.append(4)
    
        
# ここからGUI周り
root = tkinter.Tk()
root.wm_title("EXP4-AUDIO")

# 再生ボタンが押されたかどうか
is_play_running = False    

# 再生位置をテキストで表示するためのラベルを作成
text = tkinter.StringVar()
text.set('0.0')
label = tkinter.Label(master = root, textvariable=text, font=("MS Gothic", 30))
label.pack()

# 再生するファイル名
filename = 'aiueo.wav'

# pydubを使用して音声ファイルを読み込む
audio_data = AudioSegment.from_wav(filename)
p = pyaudio.PyAudio()

# pyaudioの再生ストリームを作成
p_play = pyaudio.PyAudio()
stream_play = p_play.open(
	format = p.get_format_from_width(audio_data.sample_width),	# ストリームを読み書きするときのデータ型
	channels = audio_data.channels,								# チャネル数
	rate = audio_data.frame_rate,								# サンプリングレート
	output = True												# 出力モードに設定
)

# pydubで読みこんだ音楽ファイルを再生する部分を関数化。別スレッドで実行する
def play_music():
    
    # 音楽データと再生ボタンが押されたかどうかを受け取る。また再生位置を外からも参照できるようにする
    global is_play_running, audio_data, now_playing_sec
    size_frame_music = 10 # 10ミリ秒毎に読み込む
    
    # 0.01秒ごとに再生ボタンが押されたかどうかを判定する。
    while True :
        if is_play_running:
            
            # 再生ボタンが押されたらファイルを読み込む
            idx = 0
            for chunk in make_chunks(audio_data, size_frame_music):
                
                # 再生
                stream_play.write(chunk._data)
                
                # 再生位置を計算する。
                now_playing_sec = (idx * size_frame_music) / 1000
                idx += 1
            # 一度再生されれば(stream_play.write()が実行されれば)すぐにis_play_runningをfalseにする。
            is_play_running = False
        time.sleep(0.01) # 0.01秒ごとに判定する

# 再生されているときにguiを随時更新するための関数
def update_gui():
    global is_play_running, now_playing_sec, text
    
    # 0.01ごとに再生ボタンが押されているかどうか判定し、guiを更新する
    while True:
        if is_play_running:
            
            # 再生位置のテキスト表示の更新
            text.set('%.3f' % now_playing_sec)
            
            # 再生位置のスペクトラムを表示
            _draw_spectrum(now_playing_sec)
            
            # 再生位置をスペクトログラム、識別のグラフに表示する
            l1, l2 = _draw_playing_time(now_playing_sec)
            # 再生位置のグラフが表示されたらすぐ削除する。(これにより、x = now_playing_secのグラフが横移動しているように見える)
            l1.remove() 
            l2.remove()
            
        time.sleep(0.01)

# スペクトログラム、識別のグラフに再生位置のアニメーションをつけるための関数
def _draw_playing_time(sec):
    
    # x = secの関数をそれぞれのグラフに表示
    l1 = ax2.axvline(x=sec, c = 'k')
    l2 = ax5.axvline(x=sec, c = 'k')
    canvas.draw()
    canvas3.draw()
    return l1, l2

# 再生位置は最初0に
now_playing_sec = 0.0

# 音声を別スレッドで開始
t_play_music = threading.Thread(target=play_music)
t_play_music.setDaemon(True)	# GUIが消されたときにこの別スレッドの処理も終了されるようにするため
t_play_music.start()

# 再生位置によるGUIの更新を別スレッドで開始
t_update_gui = threading.Thread(target=update_gui)
t_update_gui.setDaemon(True)	# GUIが消されたときにこの別スレッドの処理も終了されるようにするため
t_update_gui.start()

# 最初再生ボタンが押されたかどうかのフラグはfalse
is_play_running = False

# 再生ボタンが押されたら is_play_runningをtrueに
def push1():
    global is_play_running
    is_play_running = True
# 再生ボタンを設置
button1 = tkinter.Button(root, text = '再生' , command = push1, font = ("MS Gothic", 60)).pack()

# スペクトログラム、スペクトル、母音識別のグラフを作る
frame1 = tkinter.Frame(root) #スペクトログラム
frame2 =tkinter.Frame(root) #スペクトル
frame3 =tkinter.Frame(root) #母音識別
frame1.pack(side="left")
frame2.pack(side="left")
frame3.pack(side="left")

#まずスペクトログラムについて
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master = frame1)

#ラベル
ax.set_xlabel('sec')
ax.set_ylabel('frequency [Hz]')

#画像として認識させるための処理
ax.imshow(
    np.flipud(np.array(spectrogram).T),
    extent=[0,duration,0,8000],
    aspect = 'auto',
    interpolation = 'nearest'
)

#x軸のデータを作成、単位は秒(s)
x_data = np.linspace(0, duration, len(funfreq))

#基本周波数をプロットする
ax.plot(x_data, funfreq, c = 'b', label = "基本周波数")

#タイトル
ax.set_title('スペクトログラム', fontname = 'MS Gothic')

#スペクトログラムに音量データを重ねる

ax2 = ax.twinx() #2軸にする

#ラベル
ax2.set_ylabel('volume [dB]')

# x軸のデータを作成
x_data3 = np.linspace(0, duration, len(volume))

#音量データをプロット
ax2.plot(x_data3, volume, c = 'y', label = "音量")

# 基本周波数、音量のグラフのラベルをキレイに表示させるための処理
handler1, label1 = ax.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()
ax.legend(handler1 + handler2,label1 + label2, prop={"family":"MS Gothic"})

canvas.get_tk_widget().pack(side="left")


#ここから音声認識のグラフ
fig3, ax5 = plt.subplots()
canvas3 = FigureCanvasTkAgg(fig3, master=frame3)

#ラベル
plt.xlabel('sec')
plt.ylabel('recognition')

#x軸の範囲
ax5.set_xlim(0,duration)

#y軸を[0-4]から[あ、い、う、え、お]にする
ax5.set_yticks([0,1,2,3,4])
ax5.set_yticklabels(['あ','い','う', 'え','お'], fontname='MS Gothic')

#タイトル
ax5.set_title('母音識別', fontname = 'MS Gothic')

#識別データをプロット
ax5.plot(x_data, recognition)
canvas3.get_tk_widget().pack(side="left")	

#スペクトル表示するためのコールバック関数
def _draw_spectrum(v):
    
    #受け取った時間のスペクトルをスペクトログラムから切り出す
    index = int((len(spectrogram) - 1) * (float(v) / duration))
    spectrum = spectrogram[index]
    
    #直前のグラフを消去
    plt.cla()
    
    #x軸のデータを作成
    x_data = np.linspace(0, SR/2, len(spectrum))
    
    #スペクトルをプロット
    ax4.plot(x_data, spectrum)
    
    #範囲を設定
    ax4.set_ylim(-10, 5)
    ax4.set_xlim(0, SR/2)
    
    #ラベル
    ax4.set_ylabel('amblitude')
    ax4.set_xlabel('frequency [Hz]')
    
    #タイトル
    ax4.set_title('スペクトル', fontname = 'MS Gothic')
    canvas2.draw()
    
fig2, ax4 = plt.subplots()

canvas2 = FigureCanvasTkAgg(fig2, master = frame2)
canvas2.get_tk_widget().pack(side = "top")

# スライドバー
scale = tkinter.Scale(
    command=_draw_spectrum,                 # コールバック関数
    master = frame2,                        # 表示するフレーム
    from_ = 0,                              # 最小値
    to = duration,                          # 最大値
    resolution = size_shift / SR,           # 刻み幅
    label = u'スペクトルを表示する時間[sec]', # ラベル
    orient = tkinter.HORIZONTAL,            # 横方向にスライド
    length = 600,                           # 横サイズ
    width = 50,                             # 縦サイズ
    font = ("", 20)                         # フォントサイズ
)
scale.pack(side = "top")

#実行
tkinter.mainloop()

# 再生のフラグは最初false
# is_play_running = False

# # 終了処理
# stream_play.stop_stream()
# stream_play.close()
# p_play.terminate()
