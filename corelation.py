import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Audio file path
'''
MacOs path: /Users/noelalben/Downloads/BetterLookBack_Lucious_real_a.wav
Windows Os path: r'C:\path\to\your\audio.wav'
'''

audio_file = '/Users/jameswang/workspace/Audio Content Analysis/CorrelationActivity_export/Ihavecosof10s100s&1000sHz.wav' # Replace with the actual file path
# Import and load audio file
sample_rate, audio_data = wavfile.read(audio_file)
# Convert to mono if stereo
if len(audio_data.shape) > 1:
  audio_data = np.mean(audio_data, axis=1)
# Normalize amplitude
audio_data = audio_data/max(audio_data)

# Parameters
sr = 44100  # Standard audio sample rate (Hz)
duration = 1.0       # Duration in s
# Create time array
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

func = []
corr = []
for k,i in enumerate(range(2,2000)):
    phase = (i * np.pi) / 2
    func.append(np.cos(2 * np.pi * i*10 * t + phase))
    corr.append(np.dot(func[k],audio_data))


print(corr)
plt.stem(corr)
# plt.title('440 Hz Sine Wave')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
plt.grid(True)
plt.show()



