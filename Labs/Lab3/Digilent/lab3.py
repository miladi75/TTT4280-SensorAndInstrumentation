import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
#import import_data

def raspi_import(path, channels=5):


    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype=np.uint16)
        data = data.reshape((-1, channels))
    return sample_period, data


[Ts, data] = raspi_import("adcDatan.bin")





def measureSpeed(data,Fs,realChannel,imagChannel,f0=24.13e9):
    fft,freqs = complexFFT(data,Fs,realChannel,imagChannel)
    fft, freqs = bandPass(fft, freqs)
    peakFreq = findPeak(fft,freqs)
    speed = dopplerFormula(peakFreq)
    return speed


def complexFFT(data,Fs,realChannel,imagChannel):
    IFI = data[:,realChannel]
    IFQ = data[:,imagChannel]
    if(len(IFI) % 2 == 1):
        IFI = IFI[:-1]
        IFQ = IFQ[:-1]
    #plt.plot(IFQ)
    #plt.show()
    IF = IFI + 1j*IFQ
    IF = signal.detrend(IF,type='constant')
    fft = np.fft.fft(IF) 
    N = len(fft)
    temp1 = fft[N//2:]
    temp2 = fft[:N//2]
    fft[:N//2] = temp1
    fft[N//2:] = np.flip(temp2,0)
    #fft = np.concatenate(temp1,temp2)
    freqs = np.linspace(-Fs/2,Fs/2,N) 
    return fft,freqs

def findPeak(fft,freqs):
    i = np.argmax(fft)
    peakFreq = freqs[i]
    return peakFreq

def dopplerFormula(fd, f0=24.13e9):
    c=3e8
    v=c*fd/2/f0
    return v

def bandPass(fft,freqs, lowF=3, highF=500):
    compLow = np.abs(freqs - lowF)
    compHigh = np.abs(freqs - highF)
    lowIdx = np.argmin(compLow)
    highIdx = np.argmin(compHigh)
    fft = fft[lowIdx:highIdx]
    freqs = freqs[lowIdx:highIdx]
    return fft, freqs


def test(path,realChannel,imagChannel, Nmeasurments):
    Ts, data = raspi_import(path)
    
    Fs = 1e6/Ts
    
    if(Nmeasurments == 1):        
        fft,freqs = complexFFT(data,Fs,realChannel,imagChannel)
        fft, freqs = bandPass(fft,freqs)
        plt.plot(freqs,np.abs(fft))
        plt.title("DFT")
        plt.xlabel("Hz")
        plt.show()
        peak = findPeak(fft,freqs)
        return dopplerFormula(peak), peak
    
    
    datas = np.split(data, Nmeasurments)
    speeds = np.zeros(Nmeasurments)
    for i in range(Nmeasurments):
        speeds[i] = measureSpeed(datas[i], Fs, realChannel, imagChannel)
        
        """fft,freqs = complexFFT(datas[i],Fs,realChannel,imagChannel)
        fft, freqs = bandPass(fft,freqs)
        N = len(fft)
        plt.plot(freqs,np.abs(fft))
        plt.show()"""
    
  
    
    plt.plot(speeds)
    plt.title("SPEED")
    plt.show()
    speed = np.mean(speeds)
    return speed
    

speed = test("adcDatan.bin", 1,2, 50)

print("SPEED = {}".format(speed)," m/s")
    
    
