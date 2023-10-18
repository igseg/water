import pandas as pd
import numpy  as np

import matplotlib.pyplot as plt
import scipy.fft
from data_tools import *

ts = load_data()

plt.figure(1, figsize=(12,13))

plt.suptitle('Amplitude and frequency of Fourier Transform \n for different variables', fontsize= 28)

plt.subplot(221)
FFT = scipy.fft.fft(ts.evaporation.values)
FFT_abs = np.abs(FFT)
new_N = len(FFT)
f_nat = 1
new_X = 1/np.arange(0, f_nat, 1/new_N)
#FFT_abs = normalize(FFT_abs)
plt.plot(new_X,FFT_abs[0:new_N])
plt.xlabel('Period ($Month$)',fontsize=18)
plt.ylabel('Amplitude',fontsize=18)
plt.title('Evaporation',fontsize=24)
plt.grid()
plt.xlim(1.1,25)

plt.subplot(222)
FFT = scipy.fft.fft(ts.inflow.values)
FFT_abs = np.abs(FFT)
new_N = len(FFT)
f_nat = 1
new_X = 1/np.arange(0, f_nat, 1/new_N)
#FFT_abs = normalize(FFT_abs)
plt.plot(new_X,FFT_abs[0:new_N])
plt.xlabel('Period ($Month$)',fontsize=18)
plt.ylabel('Amplitude',fontsize=18)
plt.title('Water inflow',fontsize=24)
plt.grid()
plt.xlim(1.1,25)

plt.subplot(223)
FFT = scipy.fft.fft(ts.storage.values)
FFT_abs = np.abs(FFT)
new_N = len(FFT)
f_nat = 1
new_X = 1/np.arange(0, f_nat, 1/new_N)
#FFT_abs = normalize(FFT_abs)
plt.plot(new_X,FFT_abs[0:new_N])
plt.xlabel('Period ($Month$)',fontsize=18)
plt.ylabel('Amplitude',fontsize=18)
plt.title('Water inventory',fontsize=24)
plt.grid()
plt.xlim(1.1,25)

plt.subplot(224)
FFT = scipy.fft.fft(ts.price_power.values)
FFT_abs = np.abs(FFT)
new_N = len(FFT)
f_nat = 1
new_X = 1/np.arange(0, f_nat, 1/new_N)
#FFT_abs = normalize(FFT_abs)
plt.plot(new_X,FFT_abs[0:new_N])
plt.xlabel('Period ($Month$)',fontsize=18)
plt.ylabel('Amplitude',fontsize=18)
plt.title('Electricity price',fontsize=24)
plt.grid()
plt.xlim(1.1,25)

plt.savefig('../Figures/paper/fouriertransform.pdf', format = 'pdf')

plt.show()
