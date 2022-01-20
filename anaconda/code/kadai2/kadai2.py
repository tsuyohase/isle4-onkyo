# ライブラリの読み込み
from tkinter.constants import TRUE
import pyaudio
import numpy as np
import threading
import time

# matplotlib 関連
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# GUI 関連
import tkinter
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)

# mp3 ファイルを読み込んで再生
from pydub import AudioSegment
from pydub.utils import make_chunks

# 楽譜を読み込むためのライブラリ
import mido


# サンプリングレート
SAMPLING_RATE = 16000

# フレームサイズ
FRAME_SIZE = 2048

# サイズシフト
SHIFT_SIZE = int(SAMPLING_RATE/ 20)

SPECTRUM_MIN = -5
SPECTRUM_MAX = 1

# 音量を表示する際の値の範囲
VOLUME_MIN = -120
VOLUME_MAX = -10

# log10 を計算する際に、引数が0にならないようにするためにこの値を足す
EPSILON = 1e-10

# ハミング窓
hamming_window = np.hamming(FRAME_SIZE)

# グラフに表示する縦軸方向のデータ数
MAX_NUM_SPECTROGRAM = int(FRAME_SIZE / 2)

# グラフに表示する横軸方向のデータ数
NUM_DATA_SHOWN = 100

# GUI の開始フラグ
is_gui_running = False

# 楽譜のMidiファイルのデータ
TEMPO = 1071429
TICKS_PER_BEAT = 480


# 各種関数の定義

# ピークを求める関数。自己相関を求めるとき用
def is_peak(a, index):
    if(index == 0 or index == len(a) - 1) :
        return False
    else :
        return a[index - 1] < a[index] and a[index] > a[index + 1]

# ゼロ交差数を求める関数
def zero_cross_short(waveform):
    d = np.array(waveform)
    return sum([1 if x < 0.0 else 0 for x in d[1:] * d[:-1]])

# 周波数からノートナンバーへ変換
def hz2nn(frequency):
    	return int (round (12.0 * (np.log(frequency / 440.0) / np.log (2.0)))) + 69


# スペクトルの振幅をそれぞれの周波数に対応するノートナンバーに格納する
def chroma_vector(spectrum, frequencies):
    	
	cv = np.zeros(128)
	
	for s, f in zip (spectrum , frequencies):
		nn = hz2nn(f)
		cv[nn] += np.abs(s)
	
	return cv       

# ノートナンバーから周波数への変換
def nn2hz(nn):
    return int(440 * (2 ** ((nn - 69)/12)))

# ノートナンバーの候補集合にスペクトルの振幅を割り当てる。SHS用
def shs(cv,candidate):
    sh = np.zeros(len(candidate))
    for c in range(len(candidate)):
        for nc in np.arange(candidate[c], 128, 12):
            sh[c] += cv[nc]
    return sh


#GUI / グラフ描画の処理

# matplotlib animation を用いて描画する

# matplotlib animation によって呼び出される関数
def animate(frame_index):
    ax1_sub.set_array(spectrogram_data)
    
    ax2_sub.set_data(time_x_data, volume_data)
    
    ax3_sub.set_array(spectrogram_data_music)
    
    ax4_sub.set_data(time_x_data,fun_frequency_data)
    
    ax5_sub.set_data(time_x_data,fun_frequency_data_music)
    
    ax6_sub.set_data(time_x_data2,notenumber_data_music)
    
    ax7_sub.set_data(time_x_data,notenumber_data)
    
    
    return ax1_sub, ax2_sub,ax3_sub,ax4_sub,ax5_sub,ax6_sub,ax7_sub


# GUIで表示するための処理
root = tkinter.Tk()
root.wm_title("EXP4_AUDIO")

# 終了ボタンが押されたときに呼び出される関数
def _quit():
    root.quit()
    root.destroy()
    
# 終了ボタン
button = tkinter.Button(master=root, text="終了", command= _quit, font=("",30))
button.pack()

# 再生位置をテキストで表示するためのラベルを作成
text = tkinter.StringVar()
text.set('0.0')
label = tkinter.Label(master=root, textvariable=text, font=('',30))
label.pack()

# Tkinter のウィジェットを階層的に管理するためにFrameを使用
# frame1 ... マイク入力から受け取った音声データのスペクトログラムを表示
# frame2 ... 楽曲データのスペクトログラムを表示
# frame3 ... カラオケ画面を表示
frame1 = tkinter.Frame(root)
frame2 = tkinter.Frame(root)
frame3 = tkinter.Frame(root)
frame1.pack(side="left")
frame2.pack(side="left")
frame3.pack(side="top")



# 横軸の値のデータ
# time_x_data ... 音声データと楽曲データの横軸のデータ
# time_x_data2 ... 楽譜データの横軸のデータ
time_x_data = np.linspace(0, NUM_DATA_SHOWN * (SHIFT_SIZE/SAMPLING_RATE), NUM_DATA_SHOWN)
time_x_data2 = np.arange(0, NUM_DATA_SHOWN * (SHIFT_SIZE/SAMPLING_RATE),1/(TICKS_PER_BEAT * 1000000 / TEMPO))
# 縦軸の値のデータ
freq_y_data = np.linspace(8000/MAX_NUM_SPECTROGRAM, 8000, MAX_NUM_SPECTROGRAM)

# 初期値のデータを作成
# spectrogram_data ... 音声のスペクトログラムを格納するデータ
# volume_data ... 音量を格納するデータ
# fun_frequency_data ... 音声の基本周波数を格納するデータ
# fun_frequency_data_music ... 楽曲の基本周波数を格納するデータ
# notenumber_data ... 音声のノートナンバーを格納するデータ
# notenumber_data_music ... 楽譜のノートナンバーを格納するデータ
# spectrogram_data_music ... 楽曲のスペクトログラムを格納するデータ
spectrogram_data = np.zeros((len(freq_y_data), len(time_x_data)))
volume_data = np.zeros(len(time_x_data))
fun_frequency_data = np.zeros(len(time_x_data))
fun_frequency_data_music = np.zeros(len(time_x_data))
notenumber_data = np.zeros(len(time_x_data))
notenumber_data_music = np.zeros(len(time_x_data2))
spectrogram_data_music = np.zeros((len(freq_y_data),len(time_x_data)))
    
# 音声データのスペクトログラムを描画
fig, ax1 = plt.subplots(1,1,figsize = (3,3))
canvas = FigureCanvasTkAgg(fig, master = frame1)

# 音声データのスペクトルグラムを描写するために行列に
X = np.zeros(spectrogram_data.shape)
Y = np.zeros(spectrogram_data.shape)
for idx_f, f_v in enumerate(freq_y_data):
    for idx_t, t_v in enumerate(time_x_data):
        X[idx_f, idx_t] = t_v
        Y[idx_f, idx_t] = f_v

# pcolormeshを用いてスペクトログラムを描写
ax1_sub = ax1.pcolormesh(
    X,
    Y,
    spectrogram_data,
    shading='nearest',
    cmap='jet',
    norm=Normalize(SPECTRUM_MIN, SPECTRUM_MAX)
)    

# 音量を表示するために反転した軸を作成
ax2 = ax1.twinx()

# 音量をプロット
ax2_sub, = ax2.plot(time_x_data, volume_data, c='k')

# 基本周波数をプロット
ax4_sub, = ax1.plot(time_x_data,fun_frequency_data, c='b')

# ラベルの設定
ax1.set_xlabel('sec')
ax1.set_ylabel('frequency [Hz]')
ax2.set_ylabel('volume [dB]')
ax1.set_title('音声データ', fontname = 'MS Gothic')

# 音量を表示する際の値の範囲を設定
ax2.set_ylim([VOLUME_MIN, VOLUME_MAX])


# matplotlib animationを設定
ani = animation.FuncAnimation(
    fig,
    animate,
    interval=500,
    blit=True
)

# matplotlib を GUIに追加
toolbar = NavigationToolbar2Tk(canvas, frame1)
canvas.get_tk_widget().pack()

# 楽曲データのスペクトログラムを描画
fig2, ax3 = plt.subplots(figsize = (3,3))
canvas2 = FigureCanvasTkAgg(fig2, master=frame2)

# 楽曲データの基本周波数をプロット
ax5_sub, = ax3.plot(time_x_data,fun_frequency_data_music, c='k')

# ラベルの設定
ax3.set_xlabel('sec')
ax3.set_ylabel('frequency [Hz]')
ax3.set_title('楽曲データ', fontname = 'MS Gothic')

# 楽曲データのスペクトルグラムを描写するために行列に
X2 = np.zeros(spectrogram_data_music.shape)
Y2 = np.zeros(spectrogram_data_music.shape)
for idx_f, f_v in enumerate(freq_y_data):
    for idx_t, t_v in enumerate(time_x_data):
        X2[idx_f, idx_t] = t_v
        Y2[idx_f, idx_t] = f_v

# pcolormeshを用いてスペクトログラムを描写
ax3_sub = ax3.pcolormesh(
    X2,
    Y2,
    spectrogram_data_music,
    shading='nearest',
    cmap='jet',
    norm=Normalize(-9, -4)
)    

toolbar = NavigationToolbar2Tk(canvas2, frame2)
canvas2.get_tk_widget().pack()

# カラオケ採点画面を描画
fig3, ax6 = plt.subplots(figsize = (3,3))
canvas3 = FigureCanvasTkAgg(fig3, master = frame3)

# 楽譜データから楽曲の正しいノートナンバー(音高)をプロット
ax6_sub, = ax6.plot(time_x_data2,notenumber_data_music, lw=3)

# 音声データのノートナンバーをプロット
ax7_sub, = ax6.plot(time_x_data,notenumber_data)

# ラベルを設定
ax6.set_xlabel('sec')
ax6.set_ylabel('note_number')
ax6.set_title('カラオケ採点', fontname = 'MS Gothic')

# 縦軸の表示範囲を制限
ax6.set_ylim([40,70])

toolbar = NavigationToolbar2Tk(canvas3, frame3)
canvas3.get_tk_widget().pack()

# 歌詞を表示するためのラベルを作成
lyric = tkinter.StringVar()
lyric.set('')
lyrics = tkinter.Label(master=root, textvariable=lyric, font=('',30))
lyrics.pack()

# 採点結果を表示するためのラベルを作成
score = tkinter.StringVar()
score.set('100')
score_board = tkinter.Label(master=root, textvariable=score, font=('',30))
score_board.pack()



# マイク入力のための処理

x_stacked_data = np.array([])

# ノートナンバーの候補
candidate = range(55,76)

# 音声データのノートナンバー全体を格納する(採点機能用)
notenumber_data_whole =[]

def input_callback(in_data, frame_count, time_info, status_flags):
    
    # この関数は別スレッドで実行するため
    # メインスレッドで定義したものを利用できるように global 宣言する
    # x_stacked_data ... フレーム毎の音声データ
    # spectrogram_data ... フレーム毎の音声データのスペクトログラム
    # volume_data ... 表示する音量データ
    # fun_frequency_data ... 表示する音声の基本周波数のデータ
    # notenumber_data ... 表示する音声のノートナンバーのデータ
    global x_stacked_data, spectrogram_data, volume_data, fun_frequency_data,notenumber_data
    
    # 現在のフレームの音声データをnumpy array に変換
    x_current_frame = np.frombuffer(in_data, dtype=np.float32)
    
    # 現在のフレームとこれまでに入力されたフレームを連結
    x_stacked_data = np.concatenate([x_stacked_data,x_current_frame])
    
    # フレームサイズ分のデータがあれば処理を行う
    if len(x_stacked_data) >= FRAME_SIZE:
        
        # フレームサイズからはみ出した過去のデータは捨てる
        x_stacked_data = x_stacked_data[len(x_stacked_data) - FRAME_SIZE:]
        
        # スペクトルを計算
        fft_spec = np.fft.rfft(x_stacked_data *hamming_window)
        fft_log_abs_spec = np.log10(np.abs(fft_spec) + EPSILON)[:-1]
        
        # ２次元配列上で列方向（時間軸方向）に１つずらし（戻し）
        # 最後の列（＝最後の時刻のスペクトルがあった位置）に最新のスペクトルデータを挿入
        spectrogram_data = np.roll(spectrogram_data, -1,axis=1)
        spectrogram_data[:, -1] = fft_log_abs_spec
        
        # 音量も同様の処理
        vol = 20 * np.log10(np.mean(x_current_frame ** 2) + EPSILON)
        volume_data = np.roll(volume_data, -1)
        volume_data[-1] = vol
        
        
        # 音声データの基本周波数を自己相関を用いて求める
        
        # 自己相関を求め、
        autocorr = np.correlate(x_stacked_data, x_stacked_data, 'full')
        
        # 後半は必要ないので捨てる
        autocorr = autocorr [len(autocorr) // 2:]
        
        # 0でないピークとなるインデックスを求める
        peakindices = [i for i in range (len (autocorr)) if is_peak (autocorr, i)]
        peakindices = [i for i in peakindices if i != 0]
        
        # 最大となるインデックスを求める
        max_peak_index = max(peakindices , key=lambda index: autocorr [index], default=1)
        
        # 逆数が基本周波数
        ff = SAMPLING_RATE/max_peak_index
        
        # ゼロ交差数による有声音、無声音の推定を行う。
        zero_cross_number = zero_cross_short(x_stacked_data) * SAMPLING_RATE // FRAME_SIZE
        
        # スペクトログラム、音量と同じ処理
        
        # 基本周波数のデータ
        fun_frequency_data = np.roll(fun_frequency_data, -1)
        
        # ノートナンバーのデータ
        notenumber_data = np.roll(notenumber_data, -1)
        
        # ゼロ交差数が基本周波数の2倍程度ならそのまま格納、それ以外なら格納しない。
        if(ff * 8 > zero_cross_number and ff * 1 < zero_cross_number):
            # 基本周波数
            fun_frequency_data[-1] = ff
            
            # ノートナンバー
            notenumber_data[-1] = hz2nn(ff)
            
            # 全体のノートナンバー(フレームが進むごとにデータを捨てることなく格納していく)
            notenumber_data_whole.append(hz2nn(ff))
        else :
            # 無声音なら格納しない
            fun_frequency_data[-1] = None
            notenumber_data[-1] = None
            notenumber_data_whole.append(0)
        
    return None, pyaudio.paContinue

# マイクからの音声入力にはpyaudioを使用
p = pyaudio.PyAudio()
stream = p.open(
    format = pyaudio.paFloat32,
    channels = 1,
    rate = SAMPLING_RATE,
    input = True,
    frames_per_buffer = SHIFT_SIZE,
    stream_callback = input_callback
)

# mp3ファイル音楽を再生する処理

# mp3ファイル名
filename = 'ao.mp3'

# pydubを使用して音楽ファイルを読み込む
audio_data = AudioSegment.from_mp3(filename)

# 音声ファイルの再生にはpyaudioを使用
# ここではpyaudioの再生ストリームを作成
p_play = pyaudio.PyAudio()
stream_play = p_play.open(
    format = p.get_format_from_width(audio_data.sample_width),
    channels= audio_data.channels,
    rate = audio_data.frame_rate,
    output = True
)


# 楽譜データから楽曲の正しいノートナンバー(音高)を読み取る

# midiファイルの読み込み
mid = mido.MidiFile('aogeba.mid')

# 歌詞を定義、[歌詞、表示タイミング]の配列
lyrics_set = [['仰げば尊し 我が師の恩',None],['教えの庭にも はや幾年',None],['思えばいととし この年月',None],['今こそ別れめ いざさらば',None]]

# 楽曲の正しいノートナンバー
melody = []

# 楽譜のすべての音符のノートナンバーと開始位置、終了位置を格納する配列
melody_note = [[0] * 3 for i in range(100)]

# 読み込んだ位置
play_time = 0

# 歌詞番号(0-3)
lyrics_num = 0

# 読み込んだ音符の数
melody_note_num = 0

# midi ファイルを読み込む
for i,tr in enumerate(mid.tracks):
    preNote = None
    for msg in tr:
        # ノート(音符)を表すデータのときのみ処理を行う
        if msg.type == 'note_on':
            # チャンネル0のみ取り込む
            if msg.channel == 0:
                # 読み込んだ時間を加算
                play_time +=msg.time
                # velocity (音量) が0より大きいとき、音符を表す
                if msg.velocity > 0:
                    melodysub = [preNote] * msg.time
                    melody.extend(melodysub)
                    preNote = msg.note-12
                else: # velocity が0のとき、休符を表す
                    melodysub = [preNote] * msg.time
                    melody.extend(melodysub)
                    preNote = None
                    melody_note[melody_note_num][0] = msg.note
                    melody_note[melody_note_num][1] = play_time - msg.time
                    melody_note[melody_note_num][2] = play_time
                    melody_note_num +=1
                # time が1000 以上(長い休符)があるとき、歌詞の途切れめとして歌詞番号に1加算
                if msg.time > 1000:
                    if lyrics_num <4:
                        lyrics_set[lyrics_num][1] = play_time
                        lyrics_num +=1
        else:
            print(msg)
# マイク入力とのタイミングをあわせるため微調整     
melody = np.roll(melody,360)

# 楽譜データが入っていない部分は捨てる
melody_note =[i for i in melody_note if i[0] >0]


# 楽曲のデータを格納
x_stacked_data_music = np.array([])

# pydubで読み込んだ音楽ファイルを再生する部分のみ関数化
def play_music():
    
    # この関数は別スレッドで実行するため
    # メインスレッドで定義したものを利用できるように global 宣言する
    
    # is_gui_running ... GUIが動作しているかどうか
    # audio_data ... 音楽ファイル
    # now_playing_sec ... 楽曲の再生位置
    # x_stacked_data_music ... フレーム毎の楽曲データ
    # spectrogram_data_music ... フレーム毎の楽曲データのスペクトログラム
    # fun_frequency_data_music ... 表示する楽曲データの基本周波数の配列
    # melody ... 楽曲の正しいノートナンバー
    # notenumer_data_music ... 表示する楽曲データの正しいノートナンバーの配列
    # lyrics_set ... 歌詞と表示タイミングのデータ
    # lyric ... 表示する歌詞
    # notenumber_data_whole ... 音声データのノートナンバー
    # score ... 採点結果を格納する変数  
    global is_gui_running, audio_data, now_playing_sec, x_stacked_data_music, spectrogram_data_music,fun_frequency_data_music,melody,notenumber_data_music,lyrics_set,lyric,notenumber_data_whole,score
    
    # pydub のmake_chunksを用いて音楽ファイルのデータを切り出しながら読み込む
    
    size_frame_music = 25  # 25 ミリ秒毎に読み込む
    
    idx = 0
    
    # 読み込んだ歌詞番号
    lyrics_num = 0
    
    # 音声データのノートナンバーの横軸間隔
    time_data_space = SHIFT_SIZE/SAMPLING_RATE
    
    # 楽譜データの正しいノートの横軸間隔
    tick_data_space = TEMPO / (TICKS_PER_BEAT * 1000000)
    
    # 現在の採点結果を格納する変数
    live_score = 100
    
    # 採点対象の音符の番号
    k=0

    # make_chunks関数を使用して一定のフレーム毎に音楽ファイルを読み込む
    for chunk in make_chunks(audio_data, size_frame_music):
        
        # GUIを終了していれば、この関数の処理も終了する
        if is_gui_running == False:
            break
        
        # それぞれの音符ごとに正しい音高とマイク入力からの音高の誤差をもとめ、採点する。
        
        # マイク入力により格納された音声データの大きさ
        note_data_num = len(notenumber_data_whole)
        
        # 何秒間分のデータか
        note_data_time = note_data_num * time_data_space
        
        # 何tick分のデータか
        note_data_tick = int(note_data_time /tick_data_space)
        
        # 最後の音符まで採点したら終了
        if k < len(melody_note):
            # マイク入力からの音声データの位置がそれぞれの音符の位置まで来ると採点する。
            if melody_note[k][2] < note_data_tick -1000:
                correct_note = melody_note[k][0] -12    # 正しい音程
                start_ticks = melody_note[k][1]    # 音符の開始位置
                end_ticks = melody_note[k][2]    # 音符の終了位置
                start_time_data = int(start_ticks * tick_data_space/time_data_space) + 22  # 音符の開始位置を音声データと対応付ける
                end_time_data = int(end_ticks * tick_data_space/time_data_space) + 22    # 音符の終了位置を音声データと対応付ける
                
                # 採点対象の音符に対応する音声データを取り出す
                scored_data = notenumber_data_whole[start_time_data:end_time_data]  
                
                # ノートナンバーが10以上正解と違う場合、音高の認識ミスとして採点対象から外す  
                scored_data = [l for l in scored_data if np.abs(l-correct_note) < 10]
                
                # 採点対象の音声データのノートナンバーの平均を求める
                scored_data_ave = np.average(scored_data)
                
                # 正解との誤差
                pitch_error = np.abs(scored_data_ave - correct_note)
                
                # 誤差がノートナンバー換算で1より大きければ1点減点
                if pitch_error >1:
                    live_score -= 1
                
                # 画面に表示するためラベルにセット
                score.set(live_score)
                
                # 次の音符を採点するために加算
                k +=1
        else :
            break
        
            
        # pyaudioの再生ストリームに切り出した音楽データを流し込む
        stream_play.write(chunk._data)
        
        # 現在の再生位置を計算
        now_playing_sec = (idx * size_frame_music)
        
        # 1フレームあたり何tick進むか
        ticks2second = 1000 * size_frame_music * TICKS_PER_BEAT/TEMPO
        
        # 1フレームあたりticks2second分だけnotenumber_data_musicをずらす
        for i in range(int(ticks2second)):
            notenumber_data_music = np.roll(notenumber_data_music, -1)
            if int(ticks2second * idx) + i -1 <len(melody):
                notenumber_data_music[-1] = melody[int(ticks2second * idx) + i-1] 
            
       
        # 歌詞の表示タイミングが来たら歌詞を表示、更新する
        if ticks2second * idx > lyrics_set[lyrics_num][1]:
            lyric.set(lyrics_set[lyrics_num][0])
            if lyrics_num <3 :
                lyrics_num +=1
        
        
        idx += 1
        
        # データの取得
        data_music = np.array(chunk.get_array_of_samples())
        
        # 正規化
        data_music = data_music / np.iinfo(np.int32).max
        
        # 現在のフレームとこればでに入力されたフレームを連結
        x_stacked_data_music = np.concatenate([x_stacked_data_music, data_music])
        
        # フレームサイズ分のデータがあれば処理を行う
        if len(x_stacked_data_music) >= FRAME_SIZE:
            
            # フレームサイズからはみ出した過去のデータは捨てる
            x_stacked_data_music = x_stacked_data_music[len(x_stacked_data_music)- FRAME_SIZE:]
            
            # スペクトルを計算
            fft_spec = np.fft.rfft(x_stacked_data_music * hamming_window)
            fft_log_abs_spec = np.log10(np.abs(fft_spec) + EPSILON)[:-1]
            
            # マイクの入力のときと同じ処理
            spectrogram_data_music = np.roll(spectrogram_data_music, -1, axis = 1)
            spectrogram_data_music[:, -1] = fft_log_abs_spec
            
            # frequencies = np.linspace(8000/len(fft_spec), 8000, len(fft_spec))
            # s = chroma_vector(fft_spec,frequencies)
            # sh = shs(s,candidate) 
            # max_index = np.argmax(sh) +candidate[0]
            # fre = nn2hz(max_index)
            
            # 自己相関を用いて楽曲の基本周波数を求める。マイク入力のときと同じ処理
            autocorr = np.correlate(x_stacked_data_music, x_stacked_data_music, 'full')
            autocorr = autocorr [len(autocorr) // 2:]
            peakindices = [i for i in range (len (autocorr)) if is_peak (autocorr, i)]
            peakindices = [i for i in peakindices if i != 0]
            peakindices = [i for i in peakindices if i > 20]
            max_peak_index = max(peakindices , key=lambda index: autocorr [index], default=1)
            fre = SAMPLING_RATE/max_peak_index
            fun_frequency_data_music = np.roll(fun_frequency_data_music, -1)
            fun_frequency_data_music[-1] = fre
   
                
            
# 再生時間の表示を随時更新する関数
def update_gui_text():
    global is_gui_running, now_playing_sec, text
    
    while True:
        
        # GUIが表示されていれば再生位置をテキストとして表示
        if is_gui_running:
            text.set('%.3f' % now_playing_sec)
            
        # 0.01秒ごとに更新
        time.sleep(0.01)
        
# 再生時間を表す
now_playing_sec = 0.0

# 音楽を再生するパートを別スレッドで再生開始
t_play_music = threading.Thread(target=play_music)
t_play_music.setDaemon(True)
t_play_music.start()

# 再生時間の表示を随時更新する関数を別スレッドで開始
t_update_gui = threading.Thread(target=update_gui_text)
t_update_gui.setDaemon(True)
t_update_gui.start()


# 全体の処理を実行

# GUIの開始フラグをTrueに
is_gui_running = True

# GUIを開始
tkinter.mainloop()



# GUIの開始フラグをFalseに
is_gui_running = False

# 終了処理
stream_play.stop_stream()
stream_play.close()
p_play.terminate()