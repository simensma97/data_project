import os
import numpy as np
import wave


class data:
    fr = 44100
    max = 500

    def __init__(self, tune):
      self.dir = './chords'
      self.path = os.path.join(self.dir, tune)
      self.list_freq = []
      self.freq_pres = []
      self.chords_pres = []

      self.start = 0


      spf = wave.open(self.path, 'r')
      self.nr = spf.getnframes()
      #print(self.nr)
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
        #print(np.max(self.freq_array))

      self.freq_magnitude = abs(self.freq_magnitude)
      self.freq_magnitude = self.freq_magnitude / np.sum(self.freq_magnitude)
      
      index = np.argmax(self.freq_magnitude)
      self.freq = self.freq_array[index]

      return self.freq, self.freq_array, self.freq_magnitude
    
    #This part is to get multiple frequencies at once.

    def find_max(self, max):
      for i in range(self.start,len(self.list_freq), 1):

        if self.list_freq[i] < max+4 and self.list_freq[i] > max-4:

          None

        elif self.list_freq[i] < (2*max)+4 and self.list_freq[i] > (2*max)-4:
          None

        else:
          self.freq_pres = np.append(self.freq_pres, self.list_freq[i])
          self.start = i
          break
    
    def get_chords(self):
      self.freq_array = self.fft()[1]
      self.mag_array = self.fft()[2]
      self.sort_mag = np.sort(self.mag_array)[::-1] 
      #print(self.sort_mag[:10])

      

      for i in range(len(self.sort_mag)):
        mag = self.sort_mag[i]
        ind = np.where(self.mag_array==mag)

        freq = self.freq_array[ind]
        self.list_freq = np.append(self.list_freq, freq)

      self.list_freq = np.sort(self.list_freq[:20])

      #print(self.list_freq)


      self.freq_pres = np.append(self.freq_pres, self.list_freq[0])

      self.start=1

      for i in range(2):
        self.find_max(self.freq_pres[i])


      self.freq_pres = np.sort(self.freq_pres)

      print(self.freq_pres)

      return self.freq_pres



a = data('test1.wav')
b = data.get_chords(a)

