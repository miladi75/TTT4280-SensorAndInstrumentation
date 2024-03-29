import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
#import importdata as ipd
import matte as bml
#import getangle as ga
from scipy import signal


#global for system:
x_12 = 0.065
x_13 = 0.065
x_23 = 0.06
a = 0.0562916512
f_s = 31250

channels = 5

path = 'adcData3.bin'
x = np.linspace(-np.pi, np.pi, 201)

def raspi_import(path, channels):

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype=np.uint16)
        data = data.reshape((-1, channels))
    return sample_period, data





def preProc(data, upSampleFactor):
    length = len(data[:, 0])
    mic = np.zeros((length, 5))
    
    mic = signal.detrend(data[:, :5], axis=0, type="constant")

    if (upSampleFactor != 1):
        mic = signal.resample(mic, upSampleFactor * length, axis=0)

    # Might also add filter here
    return mic

def getAngle(path):
    x_12 = 0.065
    x_13 = 0.065
    x_23 = 0.065
    a = 0.0563
    f_s = 31250

    channels = 5

    [nomTp, rawData] = raspi_import(path,channels)

    rawData_up = preProc(rawData,10)

    # Henter rekker for corsscorr
    seq1 = rawData[5:,0]
    seq2 = rawData[5:,1]
    seq3 = rawData[5:,2]

    # Fjerne DC
    seq1_no_dc = bml.removeDC(seq1)
    seq2_no_dc = bml.removeDC(seq2)
    seq3_no_dc = bml.removeDC(seq3)

    # Beregner xCorr
    corr_12 = np.correlate(seq1_no_dc,seq2_no_dc,'full')
    corr_12_abs = np.abs(np.correlate(seq1_no_dc,seq2_no_dc,'full'))
    
    corr_13 = np.correlate(seq1_no_dc,seq3_no_dc,'full')
    corr_23 = np.correlate(seq2_no_dc,seq3_no_dc,'full')

    # Alle autocorr har samme midtverdi
    corr_23_short = corr_23[31144:31344]
    corr_12_short = corr_12[31144:31344]
    corr_13_short = corr_13[31144:31344]

    # Python klarer ikke indexer på numpy lister tydeligvis. Lager nye identiske "vanlige" lister.
    corr_23_short_mod = []
    for sample in corr_23_short:
        corr_23_short_mod.append(sample)

    corr_13_short_mod = []
    for sample in corr_13_short:
        corr_13_short_mod.append(sample)

    corr_12_short_mod = []
    for sample in corr_12_short:
        corr_12_short_mod.append(sample)

    n_23 = corr_23_short_mod.index(max(corr_23_short_mod)) - (len(corr_23_short_mod))/2
    n_13 = corr_13_short_mod.index(max(corr_13_short_mod)) - (len(corr_13_short_mod))/2
    n_12 = corr_12_short_mod.index(max(corr_12_short_mod)) - (len(corr_12_short_mod))/2

    x_t = bml.determineX(x_12,x_13,x_23, n_12, n_13,n_23, f_s,a)
    vinkel = bml.calculateAngle(n_21 = n_12, n_31 = n_13, n_32 = n_23)

    if x_t < 0:
        vinkel = bml.calculateAngle(n_21 = n_12, n_31 = n_13, n_32 = n_23) + np.pi

    return vinkel



plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)



[nomTp, rawData] = raspi_import(path,channels)

rawData_up = preProc(rawData,5)

d1 = rawData_up[100:,0]
d2 = rawData_up[100:,1]
d3 = rawData_up[100:,2]
d4 = rawData_up[100:,3]
d5 = rawData_up[100:,4]



plt.subplot(5, 1, 1)
plt.title("Raw data from ADC 1")
plt.xlabel("Sample")
plt.ylabel("CV")
plt.plot(np.arange(len(d1))*32e-6, d1)

plt.subplot(5, 1, 2)
plt.title("Raw data from ADC 2")
plt.plot(np.arange(len(d2))*32e-6, d2)
plt.xlabel("Sample")
plt.ylabel("CV")

plt.subplot(5, 1, 3)
plt.title("Raw data from ADC 3")
plt.plot(np.arange(len(d3))*32e-6, d3)
plt.xlabel("Sample")
plt.ylabel("CV")

plt.subplot(5, 1, 4)
plt.title("Raw data from ADC 4")
plt.plot(np.arange(len(d4))*32e-6, d4)
plt.xlabel("Sample")
plt.ylabel("CV")

plt.subplot(5, 1, 5)
plt.title("Raw data from ADC 5")
plt.plot(np.arange(len(d5))*32e-6, d5)
plt.xlabel("Sample")
plt.ylabel("CV")

plt.show()

# Henter rekker for corsscorr
seq1 = rawData[5:,0]
seq2 = rawData[5:,1]
seq3 = rawData[5:,2]

# Trekke fra DC
sum_seq1 = 0
sum_seq2 = 0
sum_seq3 = 0

for sample in seq1:
    sum_seq1 += sample
for sample in seq2:
    sum_seq2 += sample
for sample in seq3:
    sum_seq3 += sample

mean_seq1 = sum_seq1 / len(seq1)
mean_seq2 = sum_seq2 / len(seq2)
mean_seq3 = sum_seq3 / len(seq3)

seq1_mod = []
seq2_mod = []
seq3_mod = []

for sample in seq1:
    new = sample - mean_seq1
    seq1_mod.append(new)
for sample in seq2:
    new = sample - mean_seq2
    seq2_mod.append(new)
for sample in seq3:
    new = sample - mean_seq3
    seq3_mod.append(new)


#new figure


# Plotter samples
plt.subplot(3, 1, 1)
plt.title("seq 1")
plt.xlabel("bla")
plt.ylabel("bla")
plt.plot(seq1_mod)

plt.subplot(3, 1, 2)
plt.title("seq 2")
plt.xlabel("bla")
plt.ylabel("bla")
plt.plot(seq2_mod)

plt.subplot(3, 1,3)
plt.title("seq 3")
plt.xlabel("bla")
plt.ylabel("bla")
plt.plot(seq3_mod)

plt.show()

# Beregner xCorr
corr_12 = np.correlate(seq1_mod,seq2_mod,'full')
corr_13 = np.correlate(seq1_mod,seq3_mod,'full')
corr_23 = np.correlate(seq2_mod,seq3_mod,'full')

# Bruker auto corr til aa finne rett time shift
corr_auto = np.correlate(seq3_mod,seq3_mod,"full")

corr_auto_mod = []
for sample in corr_auto:
    corr_auto_mod.append(sample)

m = max(corr_auto_mod)
i = corr_auto_mod.index(m)

print(i)

# Alle autocorr har samme midtverdi
corr_23_short = corr_23[31144:31344]
corr_12_short = corr_12[31144:31344]
corr_13_short = corr_13[31144:31344]

# Python klarer ikke indexer paa numpy lister tydeligvis. Lager nye identiske "vanlige" lister.
corr_23_short_mod = []
for sample in corr_23_short:
    corr_23_short_mod.append(sample)

corr_13_short_mod = []
for sample in corr_13_short:
    corr_13_short_mod.append(sample)

corr_12_short_mod = []
for sample in corr_12_short:
    corr_12_short_mod.append(sample)

#Regner ut antall samples time delay
print("*------------------------------------------*")

print("The lengt of corr_23_short is " + str(len(corr_23_short)))
print("The max value of corr_23 is "+ str(max(corr_23_short)))
print("It is located at index " + str(corr_23_short_mod.index(max(corr_23_short_mod))))
print("That corresponds to a delay of " + str(corr_23_short_mod.index(max(corr_23_short_mod)) - (len(corr_23_short_mod))/2)+ " samples.")

print("*------------------------------------------*")

print("The lengt of corr_13_short is " + str(len(corr_13_short)))
print("The max value of corr_13 is "+ str(max(corr_13_short)))
print("It is located at index " + str(corr_13_short_mod.index(max(corr_13_short_mod))))
print("That corresponds to a delay of " + str(corr_13_short_mod.index(max(corr_13_short_mod)) - (len(corr_13_short_mod))/2)+ " samples.")

print("*------------------------------------------*")

print("The lengt of corr_12_short is " + str(len(corr_12_short)))
print("The max value of corr_12 is "+ str(max(corr_12_short)))
print("It is located at index " + str(corr_12_short_mod.index(max(corr_12_short_mod))))
print("That corresponds to a delay of " + str(corr_12_short_mod.index(max(corr_12_short_mod)) - (len(corr_12_short_mod))/2)+ " samples.")

print("*------------------------------------------*")



# Plotter alle xCorr
plt.subplot(3, 1, 1)
plt.title("xCross 2_3")
plt.xlabel("bla")
plt.ylabel("bla")
plt.plot(range((-100),(-100)+len(corr_23_short)),corr_23_short)

plt.subplot(3, 1, 2)
plt.title("xCross 1_3")
plt.xlabel("bla")
plt.ylabel("bla")
plt.plot(range((-100),(-100)+len(corr_13_short)),corr_13_short)

plt.subplot(3, 1, 3)
plt.title("xCross 1_2")
plt.xlabel("bla")
plt.ylabel("bla")
plt.plot(range((-100),(-100)+len(corr_12_short)),corr_12_short)

plt.show()



#plt.subplot(2, 1, 1)
plt.title("Corr auto")
plt.xlabel("bla")
plt.ylabel("bla")
plt.plot(range((-31244),(-31244)+len(corr_auto)),corr_auto)
plt.show()


# Beregner vinkel

# Delays
n_23 = corr_23_short_mod.index(max(corr_23_short_mod)) - (len(corr_23_short_mod))/2
n_13 = corr_13_short_mod.index(max(corr_13_short_mod)) - (len(corr_13_short_mod))/2
n_12 = corr_12_short_mod.index(max(corr_12_short_mod)) - (len(corr_12_short_mod))/2

x_t = bml.determineX(x_12,x_13,x_23, n_12, n_13,n_23, f_s,a)
vinkel = bml.calculateAngle(n_21 = n_12, n_31 = n_13, n_32 = n_23)
print("x er: " + str(x_t))
print("raw vinkel: " + str(vinkel))
print("raw deg: " + str(vinkel * 180 / np.pi))

if x_t < 0:
    vinkel = bml.calculateAngle(n_21 = n_12, n_31 = n_13, n_32 = n_23) + np.pi



print("*------------------------------------------*")
print("Vinkelen er: " + str(bml.makeAnglePositive(vinkel * 180 / np.pi)) +  " grader.")
print("Vinkelen er: " +  str(vinkel) + " radianer.")


"""
# Henter vinkler fra andre tester:
v_1 = getAngle("radar0.bin")
v_2 = getAngle("radar36.bin")
v_3 = getAngle("radar72.bin")
v_4 = getAngle("radar108.bin")
v_5 = getAngle("radar144.bin")
"""

bml.circlePlot(bml.makeAnglePositive(vinkel * 180 / np.pi))
