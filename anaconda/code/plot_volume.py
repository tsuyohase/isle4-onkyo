
# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load('my_aiueo1.wav', sr=SR)

size_frame = 512

db = []

for i in np.arange(0, len(x)-size_frame, size_frame):
    idx = int(i)
    x_frame = x[idx : idx+size_frame]
    rms_frame = np.sqrt((np.sum(np.power(x_frame,2)))/size_frame)
    db_frame = 20 * np.log10(rms_frame)
    db.append(db_frame)

fig =plt.figure()

plt.xlabel('time')
plt.ylabel('volume [db]')
x_data = np.linspace(0,len(x)/16000,len(db))
plt.plot(x_data,db)
plt.show()

fig.savefig('plot-volume_my_aiueo.png')


