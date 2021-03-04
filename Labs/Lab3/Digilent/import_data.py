import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def raspi_import(path, channels=5):


    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype=np.uint16)
        data = data.reshape((-1, channels))
    return sample_period, data


[Ts, data] = raspi_import("adcDatan.bin")


ref = 4096

for i in range(2):
    org_data = data[:,i]
    timedomain_x = np.arange(0,0.033456,0.00003352304)
    timedomain_y = [org_data[j]*0.00080566406 for j in range(1000)]
    timedomain_y = timedomain_y[1:]
    plt.plot(timedomain_x, timedomain_y)
    plt.title("Sampling ADC " + str(i+1))
    plt.show()
    fourier = (abs(sci.fft(timedomain_y)))
    # plt.plot(20*np.log10(fourier))
    plt.plot((fourier))

    # plt.yscale("log")
    plt.title("FFT ADC "  + str(i+1))
    plt.show()