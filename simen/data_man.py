import os
import numpy as np
import wave
from collections import Counter


class data:

    array_chords = np.array([
                         [79.40368652	,105.6472778	,152.0782471	,202.5466919	,257.0526123	,339.8208618],
                         [86.8057251	,113.0493164	,161.4990234	,214.6591187	,271.8566895	,357.989502],
                         [91.51611328	,119.7784424	,170.9197998	,228.1173706	,288.0065918	,380.1956177],
                         [95.55358887	,127.180481	,181.0134888	,241.5756226	,305.5023193	,403.074646],
                         [100.9368896	,135.2554321	,191.1071777	,256.3796997	,323.6709595	,426.6265869]
    ])


    fr = 22050
    max = 500

    def __init__(self, tune):
      self.dir = '/content'
      self.path = os.path.join(self.dir, tune)
        
      self.list_freq = []
      self.freq_pres = []

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
    
    def get_chords(self,nr):
      self.freq_array = self.fft()[1]
      self.mag_array = self.fft()[2]
      #self.sort_mag = np.sort(self.mag_array)[::-1]

      self.new_freq = []
      self.mag_freq = []

      for i in range(len(self.freq_array)):
        freq = self.freq_array[i]
        mag = self.mag_array[i]
        if freq<np.amax(self.array_chords) or freq>np.amin(self.array_chords):
          self.new_freq = np.append(self.new_freq, freq)
          self.mag_freq = np.append(self.mag_freq, mag)

      self.sort_mag_freq = np.sort(self.mag_freq)[::-1]

      self.top_freq = []

      for i in range(100):
        index = np.where(self.mag_freq == self.sort_mag_freq[i])
        
        frequency = self.new_freq[index]
        self.top_freq = np.append(self.top_freq, int(frequency))

      nr = Counter(self.top_freq)
      list_common = nr.most_common()

      self.array_chords = self.array_chords.astype(int)

      #print(self.top_freq)

      self.chords = []
      counter = -1

      for i in range(len(list_common)):
        common = list_common[i]
        freq = common[0]

        if freq in self.array_chords:
          counter +=1

          index = np.where(self.array_chords == freq)
          #print(index)

          

          if index[1] == 0:
            print("1E", str(index[0]))
          elif index[1] ==1:
            print("2A", str(index[0]))
          elif index[1] ==2:
            print("3D", str(index[0]+1))
          elif index[1] ==3:
            print("4G", str(index[0]+1))
          elif index[1] ==4:
            print("5B", str(index[0]+1))
          elif index[1] ==5:
            print("6E", str(index[0]+1))

