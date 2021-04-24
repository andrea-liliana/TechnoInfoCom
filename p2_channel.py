import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import bitarray

from hamming import HammingCode, bin_array

RATE, WAV = scipy.io.wavfile.read('sound.wav')
# sound.wav: RIFF (little-endian) data, WAVE audio, Microsoft PCM, 8 bit, mono 11025 Hz

# PCM : https://en.wikipedia.org/wiki/Pulse-code_modulation#Modulation
# So signal around 128

# QUESTION 15

plt.plot(WAV)
plt.title("Original wave")

# QUESTION 16

""" Note that in the data, only 90 values out of the 256 possible
ones are used. So we only need to be able to represent 90
values. For that we need log_2 90 = 7 bits
"""

unique_elements, counts_elements = np.unique(WAV, return_counts=True)

plt.figure()
plt.hist(WAV, bins=256)
plt.title("Histogram of wave samples values counts")

mapping = [None] * 256
demapping = [2**7] * 256

for i, e in enumerate(unique_elements):
    mapping[e] = bin_array(i, 7)
    demapping[i] = e

bin_encoded = np.zeros(7 * len(WAV), dtype=bool)
for i in range(len(WAV)):
    bin_encoded[i*7:(i+1)*7] = mapping[WAV[i]]

print(f"Encoded sound is {len(bin_encoded)//8} bytes long, original was {len(WAV)}")

# Make sure we can decode it
decoded = np.zeros(len(WAV), dtype=np.uint8)
for i in range(len(WAV)):
    decoded[i] = demapping[np.dot(bin_encoded[i*7:(i+1)*7], np.array([64,32,16,8,4,2,1]))]

assert decoded.all() == WAV.all()

# QUESTION 17

P = 0.01
bits_to_flip = int(len(bin_encoded)*P)
print(f"Generating noise : {int(len(bin_encoded)*P)} bits will be flipped out of {len(bin_encoded)}")
indices = random.sample(range(len(bin_encoded)), bits_to_flip)
assert len(set(indices)) == bits_to_flip, "Some indices were redundant or not enough indices"
#print(list(sorted(indices)))

bin_encoded_with_error = bin_encoded.copy()
for i in indices:
    bin_encoded_with_error[i] = not bin_encoded_with_error[i]

decoded = np.zeros(len(WAV), dtype=np.uint8)
for i in range(len(WAV)):
    v = np.dot(bin_encoded_with_error[i*7:(i+1)*7], np.array([64,32,16,8,4,2,1]))
    decoded[i] = demapping[v]

scipy.io.wavfile.write("decoded_with_errors.wav", RATE, decoded)
print("Wrote decoded_with_errors.wav")

# QUESTION 18

DATA_BITS = 4
CODE_BITS = 7
hamming = HammingCode(DATA_BITS, CODE_BITS)

# bits = bitarray.bitarray()
# bits.frombytes(bytes(WAV.tolist()))

bits = np.append(bin_encoded, [False] * (DATA_BITS - len(bin_encoded) % DATA_BITS))
assert len(bits) % DATA_BITS == 0, "Padding is necesarry"

hamming_bits = bitarray.bitarray((len(bits)//DATA_BITS)*CODE_BITS)

print(f"Encoding {len(bits)} bits to {len(hamming_bits)} bits")
j = 0
for i in range(0, len(bits), DATA_BITS):
    data = bits[i:i+DATA_BITS].tolist()
    hcode = hamming.encode(data)
    hamming_bits[j:j+CODE_BITS] = bitarray.bitarray(list(hcode))
    j += CODE_BITS

# QUESTION 19

NB_ERRORS = int(len(hamming_bits)*P)
print(f"Introducing {NB_ERRORS} errors with probability:{P}")
indices = random.sample(range(len(hamming_bits)), NB_ERRORS)
for i in indices:
    hamming_bits[i] = hamming_bits[i] ^ 1

print("Decoding and error-correction")
decoded_bits = bitarray.bitarray()
for i in range(0, len(hamming_bits), CODE_BITS):
    hcode = hamming_bits[i:i+CODE_BITS]
    code = hamming.decode(hcode.tolist())
    decoded_bits.extend(bitarray.bitarray(list(code)))

assert len(bits) == len(decoded_bits),\
    f"The decoded data has not the same length as the original data! {len(decoded_bits)} != {len(bits)}"

error = 0
for i in range(len(bits)):
    if bits[i] != decoded_bits[i]:
        error += 1

print(f"{error} errors on {len(bits)} => {error/len(bits):.3f}")

decoded = np.zeros(len(WAV), dtype=np.uint8)
for i in range(len(WAV)):
    v = np.dot(decoded_bits[i*7:(i+1)*7].tolist(),
               np.array([64, 32, 16, 8, 4, 2, 1]))
    decoded[i] = demapping[v]

scipy.io.wavfile.write("decoded_corrected.wav", RATE, decoded)
print("Wrote decoded_corrected.wav")

print(np.min(decoded), np.max(decoded))
plt.figure()
plt.plot(decoded)
plt.title("Decoded wave, no error correction, with noise")
plt.show()

# QUESTION 20
""" Q20. How would you proceed to reduce the loss of information and/or
to improve the communication rate? Justify. """

"""

We can use longer Hamming code. The rate of information transfer
of (7,4) Hamming code is 4/7=0.57. If we use Hamming code (15,11)
for example, we have a better rate of information transfer : 11/15 = 0.73
which is higher than the (7,4) code, so it can transmit more information
through each code. That's an application of channel coding (we use longer blocks)
and it is valid because 15/11 is still lower than C = 0.919). FIXME not sure.

So to transmit n bits of information, we'll need n*7/4 bits or
n*15/11 bits. Which is shorter, so faster.

"""
