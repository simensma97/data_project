import os
import numpy as np
import wave


open_gridspec = 0
audio_test = 0

class data:
    fr = 44100
    max = 500
    
    def __init__(self, tune):
      self.dir = './chords'
      self.path = os.path.join(self.dir, tune)


      spf = wave.open(self.path, 'r')
      self.nr = spf.getnframes()
      print(self.nr)
      signal = spf.readframes(-1)

      self.signal = np.fromstring(signal, 'Int32')

    def fft(self):
      self.freq_array = np.arange(self.nr)*(float(self.fr)/self.nr) #Normaliserert den
      self.freq_array = self.freq_array[: (self.nr // 2)]  # 0 til n / 2 integrer, kutter i midten
      signal = self.signal - np.average(self.signal)  # zero-centering
      self.freq_magnitude = np.fft.fft(signal)  # fft computing and normalization
      self.freq_magnitude = self.freq_magnitude[: (self.nr // 2)]  # one side

      if self.max:
        max_index = int(self.max * self.nr / self.fr) + 1
        self.freq_array = self.freq_array[100:max_index]
        self.freq_magnitude = self.freq_magnitude[100:max_index]
        print(np.max(self.freq_array))

      self.freq_magnitude = abs(self.freq_magnitude)
      self.freq_magnitude = self.freq_magnitude / np.sum(self.freq_magnitude)
      
      index = np.argmax(self.freq_magnitude)
      self.freq = self.freq_array[index]

      return self.freq, self.freq_array, self.freq_magnitude
