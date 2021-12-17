import matplotlib.pyplot as plt
import numpy as np
import librosa
import math

def hz2nn(frequency):
    return int (round(12.0 * (math.log(frequency/ 440.0) / math.log(2.0)))) + 69

def chroma_vector_dash(spectrum, frequencies):
    cv = np.zeros(128)
    for s, f in zip (spectrum, frequencies):
        nn = hz2nn(f)
        cv[nn] += np.abs(s)
    return cv

def nn2hz(note_number):
    return 440 * (2 ** ((note_number - 69) / 12))

def subharmonic(chroma, candidate):
    shs = np.zeros(len(candidate))
    for c in range(len(candidate)):
        for nc in np.arange(candidate[c], 128, 12):
            shs[c] += chroma[nc]
    return shs

SR = 16000

x, _ = librosa.load('shs-test-man.wav', sr = SR)

spec = np.fft.rfft(x)

log_abs_spec = np.log(np.abs(spec))

size_frame = 512

hamming_window = np.hamming(size_frame)

size_shift = 16000 / 100

spectrogram = []

candidate = range(36,60)
shs_list =[]

for i in np.arange(0, len(x) - size_frame , size_shift):
    idx = int(i)
    x_frame = x[idx: idx + size_frame]
    fft_spec = np.fft.rfft(x_frame* hamming_window)
    fft_log_abs_spec = np.log(np.abs(fft_spec))
    spectrogram.append(fft_log_abs_spec)
    
    frequencies = np.linspace(8000/len(fft_spec), 8000 , len(fft_spec))
    chroma = chroma_vector_dash(fft_spec, frequencies)
    shs = subharmonic(chroma, candidate)
    max_index = np.argmax(shs) + candidate[0]
    print(max_index)
    fre = nn2hz (max_index)
    sublist = [fre] * int(size_shift)
    shs_list.extend(sublist)
    
fig = plt.figure()
plt.xlabel('sample')
plt.ylabel('frequency [Hz]')
plt.imshow(
    np.flipud(np.array(spectrogram).T),
    extent=[0, len(x), 0, SR/2],
    aspect = 'auto',
    interpolation='nearest'
)
plt.plot(shs_list)
plt.ylim([0,1000])
plt.show()

fig = plt.figure()

plt.xlabel('frequency [Hz]')	
plt.ylabel('amplitude')			
plt.xlim([0, SR/2])				
x_data = np.linspace((SR/2)/len(log_abs_spec), SR/2, len(log_abs_spec))
plt.plot(x_data, log_abs_spec)			
plt.show()
    