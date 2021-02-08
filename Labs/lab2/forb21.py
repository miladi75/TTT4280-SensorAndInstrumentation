import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

'''
def lag_finder(y1, y2, sr):
    n = len(y1)

    corr = signal.correlate(y2, y1, mode='same') / np.sqrt(signal.correlate(y1, y1, mode='same')[int(n/2)] * signal.correlate(y2, y2, mode='same')[int(n/2)])

    delay_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n)
    delay = delay_arr[np.argmax(corr)]
    print('y2 is ' + str(delay) + ' behind y1')

    plt.figure()
    plt.plot(delay_arr, corr)
    plt.title('Lag: ' + str(np.round(delay, 3)) + ' s')
    plt.xlabel('Lag')
    plt.ylabel('Correlation coeff')
    plt.show()

# Sine sample with some noise and copy to y1 and y2 with a 1-second lag
sr = 1024
y = np.linspace(0, 2*np.pi, sr)
y = np.tile(np.sin(y), 5)
y += np.random.normal(0, 5, y.shape)

y1 = y[sr:4*sr]
y2 = y[:3*sr]
plt.plot(y1, y2)
#plt.show()

#lag_finder(y1, y2, sr)
'''
def crossCorr():
    sig = np.repeat([0., 1., 1., 0., 1., 0., 0., 1.], 128)
    sig_noise = sig + np.random.randn(len(sig))
    corr = signal.correlate(sig_noise, np.ones(128), mode='full') / 128
    clock = np.arange(64, len(sig), 128)
    fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, sharex=True)
    ax_orig.plot(sig)
    ax_orig.plot(clock, sig[clock], 'ro')
    ax_orig.set_title('Original signal')
    ax_noise.plot(sig_noise)
    ax_noise.set_title('Signal with noise')
    ax_corr.plot(corr)
    ax_corr.plot(clock, corr[clock], 'ro')
    ax_corr.axhline(0.5, ls=':')
    ax_corr.set_title('Cross-correlated with rectangular pulse')
    ax_orig.margins(0, 0.1)
    fig.tight_layout()
    plt.show()

crossCorr()
