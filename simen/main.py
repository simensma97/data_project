#import matplotlib.pyplot as plt
import os
from data_man import data

path = './chords'

def tune(path):
  chords_list = os.listdir(path)
  
  freq_chords = []
  name_chords = []
  for i in range(len(chords_list)):
    if chords_list[i].endswith('.wav'):
      a = data(chords_list[i])
      freq = data.fft(a)[0]
      name = chords_list[i].split('.')[0]
    
      name_chords = np.append(name_chords, name)
      freq_chords = np.append(freq_chords, freq)
      
  return name_chords, freq_chords


tuning = tune(path)


 
