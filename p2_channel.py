from utils import *
#load_binary_text_sample(name="genome.txt",spaces=True)
# load_text_sample()
# print("--")

# import numpy as np
# f = np.genfromtxt("genome.txt",dtype='str')

# with open("genome.txt","r") as fin:
#     d = fin.read().replace("\n","")

# print(d[:100])
# exit()

import random
import math
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import scipy.io.wavfile

RATE, WAV = load_wav()

# sound.wav: RIFF (little-endian) data, WAVE audio, Microsoft PCM, 8 bit, mono 11025 Hz

# PCM : https://en.wikipedia.org/wiki/Pulse-code_modulation#Modulation
# So signal around 128

# QUESTION 15

plt.plot(WAV)

# QUESTION 16

""" Note that in the data only 90 values out of the 256 possible
ones are used. So we only need to be able to represent 90
values. For that we need log_2 90 = 7 bits

"""
unique_elements, counts_elements = np.unique(WAV, return_counts=True)

print(len(unique_elements))
print(math.ceil( math.log2(len(unique_elements))))

plt.figure()
plt.hist(WAV,bins=256)
plt.show()

mapping = [None] * 256
demapping = [128] * 256

for i, e in enumerate(unique_elements):
    mapping[e] = np.array([x == "1" for x in f"{i:7b}"])
    demapping[i] = e

bin_encoded = np.zeros(7 * len(WAV), dtype=bool)
for i in range(len(WAV)):
    bin_encoded[i*7:(i+1)*7] = mapping[WAV[i]]

print(bin_encoded)

decoded = np.zeros(len(WAV))
for i in range(len(WAV)):
    decoded[i] = demapping[np.dot(bin_encoded[i*7:(i+1)*7], np.array([64,32,16,8,4,2,1]))]

assert decoded.all() == WAV.all()

# QUESTION 17

P = 0.01

indices = random.sample(range(len(bin_encoded)), int(len(WAV)*P))
bin_encoded_with_error = bin_encoded.copy()
for i in indices:
    bin_encoded_with_error[i] = not bin_encoded_with_error[i]

decoded = np.zeros(len(WAV),dtype=np.uint8)
for i in range(len(WAV)):
    v = np.dot(bin_encoded_with_error[i*7:(i+1)*7], np.array([64,32,16,8,4,2,1]))
    decoded[i] = demapping[v]

scipy.io.wavfile.write("decoded.wav", RATE, decoded)

# QUESTION 18
