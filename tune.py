from scipy.io import wavfile
from scipy.io.wavfile import write
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
#from scipy.fft import fft, ifft
from scipy.fft import fft
from playsound import playsound
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import wave

open_gridspec = 0
audio_test = 0

class open_chords:
    def __init__(self):
        tune_dir = Path("C:\\Users\\micha\\Desktop\\Data analysis subject\\semester_project\\tunes\\open\\")
        self.c = os.listdir(tune_dir)

        self.samplerate = []
        self.data_wave = [] # Open, 0 --> A, 1 --> B, 2 --> D, 3 --> E, 4 --> e, 5 --> G (EADBGe)


        for i in enumerate(self.c):
            if i[1].endswith('.wav'):
                open_list_data = []
                open_list_sample = []
                open_path = tune_dir / i[1]
                sample, data = wavfile.read(open_path)
                open_list_data.append(data)
                open_list_sample.append(sample)
                self.data_wave.append(open_list_data)
                self.samplerate.append(open_list_sample)


        self.Open_A, self.Open_B, self.Open_D, self.Open_E, self.Open_e, self.Open_G = self.data_wave[0][0], self.data_wave[1][0], self.data_wave[2][0], self.data_wave[3][0], self.data_wave[4][0], self.data_wave[5][0]
    def time_axis(self):
        self.time = []
        for i in enumerate(self.c):
            open_array = []
            open_size = np.size(self.data_wave[i[0]][0][:,0])
            run_time = open_size/self.samplerate[i[0]][0]
            t = np.linspace(0,run_time,open_size)
            open_array.append(t)
            self.time.append(open_array)
        self.time_A, self.time_B, self.time_D, self.time_E, self.time_e, self.time_G = self.time[0][0], self.time[1][0], self.time[2][0], self.time[3][0], self.time[4][0], self.time[5][0]
    def open_grid_plot(self):
        fig = plt.figure(constrained_layout=True, figsize=(10, 10))
        widths = [2, 2, 2]
        heights = [2, 2, 2]
        gs = fig.add_gridspec(ncols=3, nrows=3, width_ratios=widths, height_ratios=heights)

        all_ax1 = fig.add_subplot(gs[0, :])
        all_ax1.plot(self.time_E, self.Open_E[:,0],'r')
        all_ax1.plot(self.time_A,self.Open_A[:,0],'b')
        all_ax1.plot(self.time_D,self.Open_D[:,0],'k')
        all_ax1.plot(self.time_G,self.Open_G[:,0],'g')
        all_ax1.plot(self.time_B,self.Open_B[:,0],'m')
        all_ax1.plot(self.time_e,self.Open_e[:,0],'y')
        all_ax1.set_title('All Open Strings')


        # Second row
        open_E_ax1 = fig.add_subplot(gs[1, 0])
        open_E_ax1.plot(self.time_E,self.Open_E[:,0],'r')
        open_E_ax1.set_title('Open E')
        open_E_ax1.set(ylabel='Magnitude')

        open_A_ax1 = fig.add_subplot(gs[1, 1])
        open_A_ax1.plot(self.time_A,self.Open_A[:,0],'b')
        open_A_ax1.set_title('Open A')


        open_D_ax1 = fig.add_subplot(gs[1, 2])
        open_D_ax1.plot(self.time_D,self.Open_D[:,0],'k')
        open_D_ax1.set_title('Open D')


        # Third row
        open_G_ax1 = fig.add_subplot(gs[2, 0])
        open_G_ax1.plot(self.time_G,self.Open_G[:,0], 'g')
        open_G_ax1.set_title('Open G')
        open_G_ax1.set(xlabel='Time (s)')
        open_G_ax1.set(ylabel='Magnitude')


        open_B_ax1 = fig.add_subplot(gs[2, 1])
        open_B_ax1.plot(self.time_B,self.Open_B[:,0], 'm')
        open_B_ax1.set_title('Open B')
        open_B_ax1.set(xlabel='Time (s)')

        open_e_ax1 = fig.add_subplot(gs[2, 2])
        open_e_ax1.plot(self.time_e,self.Open_e[:,0], 'y')
        open_e_ax1.set_title('Open e')
        open_e_ax1.set(xlabel='Time (s)')



        fig.suptitle('Open strings')

        all_ax1.legend(('E', 'A', 'D', 'G', 'B', 'e'), loc='lower center', shadow=True, ncol=3)

        plt.show()
        fig.savefig('Open_strings_time_domain.png')
    def open_play_audio(self, audio_time):
        self.audio_data_amount = audio_time*self.samplerate[4][0]
        reduced_open_E = self.Open_E[0:audio_data_amount,0]
        #print(np.shape(self.Open_E)) self.Open_E [large, 2]
        write('test_audio.wav', self.samplerate[4][0], reduced_open_E)
        playsound('C:\\Users\\micha\\Desktop\\Data analysis subject\\semester_project\\test_audio.wav')
    def fast_fourier_transform(self, audio_time):
        test = np.where((self.Open_E[:,0] > 0) & (self.Open_E[:,0] < 1000) | (np.size(self.Open_E[:,0]) <= np.size(audio_time*self.samplerate[3][0])))
        print(test[0][-1:])
        y = np.fft.fft(self.Open_E[0:1521780,0])
        # rate = self.samplerate[3][0]
        # x = np.linspace(10**7,50*10**10,y[:,0].size)
        plt.plot(y)

        #plt.plot(np.abs(y[:,0])/10**2)
        plt.xscale("log")
        #plt.yscale("log")
        #plt.ylim([10**0,10**3])
        plt.show()
    def test_string(self):
        n = 44100
        n2 = int(np.round(2.2 * n))
        point = self.Open_E[2*n:n2,0] # 0.2sec
        b = np.arange(np.size(self.Open_E[:,0]))*float(n/np.size(self.Open_E[:,0]))
        print(b[:10])
        b = b[:(np.size(self.Open_E[:,0]) // 2)]
        #point2 = self.Open_E[:,0]
        point2 = b
        point2 = point2 - np.average(self.Open_E[:,0])
        freq_mag = fft(point2)
        freq_mag = freq_mag[:(np.size(self.Open_E[:,0]) // 2)]
        plt.plot(point2, freq_mag)
        plt.xlim([0,500])
        plt.show()

        yfft = fft(point)
        yfft2 = np.abs(yfft)
        dt = np.size(point)
        x = np.linspace(0,800,dt)
        t = np.linspace(0,0.2,dt)
        x2 = np.linspace(0,800,int(dt/2))
        print(np.shape(x))
        peaks, _ = find_peaks(point,distance=150)
        print(peaks)
        difft = t[615] - t[69]
        print(difft)
        print(1/difft)
        print(point[peaks])
        # plt.plot(peaks,point[peaks],"x")
        # plt.plot(point)
        # plt.show()
        #
        # plt.semilogy(x,yfft2)
        # plt.xlim([x[0], x[-1]])
        # plt.show()
        #
        # plt.plot(x2,yfft2[0:int(np.size(point)/2)])
        # plt.yscale("log")
        # plt.xlim([0,30])
        # plt.show()
        #
        # plt.plot(t,point)
        # plt.show()
        #
        # plt.plot(np.abs(point), np.abs(yfft))
        # plt.show()

        # print(np.shape(self.Open_E))
        # # plt.plot(self.Open_E[0:10000,0])
        # # plt.show()
        # print(np.min(self.Open_E[0:10000,0]))
        # print(np.max(self.Open_E[0:10000,0]))
        # z = np.where(self.Open_E[0:10000,0] == 0)
        # test = fft(self.Open_E[2*44100:3*44100,0])
        # #T = 1/self.samplerate[3][0]
        # xf = np.linspace(0,44100,np.size(self.Open_E[2*44100:3*44100,0]))
        # print(np.size(xf))
        # peaks, _ = find_peaks(self.Open_E[2*44100:3*44100,0],distance=1)
        # print(np.size(np.abs(peaks)))
        # print(self.time_E)
        # print(np.shape(self.Open_E[2*44100:3*44100,0]))
        # attempt = np.linspace(0,1,44100)
        # T = attempt[1] - attempt[0]
        # f = np.linspace(0,1/T,np.size(self.Open_E[2*44100:3*44100,0]))
        # print(T)
        # print(1/T)
        # plt.figure(1)
        # plt.plot(self.Open_E[2*44100:3*44100,0])
        # plt.plot(peaks, self.Open_E[2*44100:3*44100,0][peaks], "x")
        #
        # plt.figure(2)
        # plt.plot(np.abs(self.Open_E[2*44100:3*44100,0]),np.abs(test))
        # plt.figure(3)
        # plt.plot(f,np.abs(test))
        # plt.show()

        #plt.plot(xf,np.abs(test))
        #plt.xscale("log")
        #plt.xlim([0,5000])
        #plt.show()
        # print(z)
        # plt.plot(self.Open_E[163:1845,0])
        # plt.plot(-self.Open_E[5314:7005,0])
        # plt.show()
    def simen_icecream(self):
        """
        Derive frequency spectrum of a signal pydub.AudioSample
        Returns an array of frequencies and an array of how prevelant that frequency is in the sample
        """

        #spf = wave.open(path, "r")
        fr = 44100
        #n = spf.getnframes()  # len of the noise
        data = self.Open_E[:,0]
        n = np.size(data)
        #n = data.getnframes()
        #signal = data.readframes(-1)
        #signal = np.fromstring(data, "Int32")
        # signal = signal_org[start:end]
        signal = data

        freq_array = np.arange(n) * (float(fr) / n)
        freq_array = freq_array[: (n // 2)]

        signal = signal - np.average(signal)

        signal = signal - np.average(signal)  # zero-centering
        #freq_magnitude = np.fft.fft(signal)  # fft computing and normalization
        freq_magnitude = fft(signal)
        freq_magnitude = freq_magnitude[: (n // 2)]  # one side

        max_frequency = 500

        if max_frequency:
            max_index = int(max_frequency * n / fr) + 1
            freq_array = freq_array[:max_index]
            freq_magnitude = freq_magnitude[:max_index]

        freq_magnitude = abs(freq_magnitude)
        freq_magnitude = freq_magnitude / np.sum(freq_magnitude)



        #max = np.argmax(freq_magnitude)
        #print(max)

        #print(freq_array[max])

        plt.plot(freq_array, freq_magnitude)
        plt.show()




Object = open_chords()
Object.time_axis()
Object.simen_icecream()
#Object.test_string()
audio_time = 5

#Object.fast_fourier_transform(audio_time=audio_time)




if audio_test == 1:
    Object.open_play_audio(audio_time=audio_time)



if open_gridspec == 1:
    Object.open_grid_plot()
#f = fft(data_wave[0])
# b=[(ele/2**16.)*2-1 for ele in data_wave[0,0:len(data_wave[0])]] # this is 8-bit track, b is now normalized on [-1,1)
# c = fft(b) # calculate fourier transform (complex numbers list)
# d = len(c)/2  # you only need half of the fft list (real signal symmetry)
#
#
#
# plt.figure(1, figsize=(10,10))
# plt.plot(data_wave[0])
# plt.figure(2, figsize=(10,10))
# plt.plot(data_wave[5])
# plt.figure(3, figsize=(10,10))
# plt.plot(abs(c[:(d-1)]),'r')
# plt.show()


