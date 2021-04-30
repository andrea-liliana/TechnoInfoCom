import math
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import bitarray

random.seed(2)

from hamming import HammingCode, bin_array

RATE, WAV = scipy.io.wavfile.read('sound.wav')
# sound.wav: RIFF (little-endian) data, WAVE audio, Microsoft PCM, 8 bit, mono 11025 Hz
# PCM : https://en.wikipedia.org/wiki/Pulse-code_modulation#Modulation

# QUESTION 15

plt.plot(WAV)
plt.title("Original wave")
plt.xlabel("Samples indices")
plt.ylabel("Samples values")
plt.savefig("q15.pdf")

# QUESTION 16

""" Note that in the data, only 90 values out of the 256 possible ones
are used. So we only need to be able to represent 90 values. For that
we need log_2 90 = 7 bits """

unique_samples, counts_samples = np.unique(WAV, return_counts=True)
# Sort samples most frquent first.
unique_samples, counts_samples = zip(
    *sorted(zip(unique_samples, counts_samples),
            key=lambda t: t[1], reverse=True))

plt.figure()
plt.hist(WAV, bins=256)
plt.title("Histogram of wave samples values counts")
plt.xlabel("Samples values")
plt.ylabel("# occurences")
plt.savefig("q16.pdf")

# Now we convert the file into a 7 bits per sample file.

# First we compute coding/decoding maps.

print(f"Q16: There are {len(unique_samples)} unique samples")
code_map = [None] * 2**8
# Note that in the decoding map, we take the
# bits that will be flipped into account
decode_map = list(range(2**7))

# There are 90 sample values. We have 128 possible codes so there are
# unused codes. We can code the first 38 samples values twice
# (occupying the the first 38*2 = 76 positions) and then the remaining
# 52 samples values once. The thing to note is that the repeated
# values each have two codes that differ by only one bit. So if the
# channel flips that bit, then the decoded value will be the same. So
# we use the unused code to build some redundancy.  Moreover, we
# associate the most frequent sample values to these repeated code,
# maximizing the efficiency of the scheme.

values_expander = [None] * 128
for i in range(38):
    values_expander[i*2] = i
    values_expander[i*2+1] = i
for i in range(38, 90):
    values_expander[38*2+i-38] = i

for i, e in enumerate(unique_samples):
    ve = values_expander.index(i)
    code_map[e] = bin_array(ve, 7)

decode_map = values_expander
decode_map = [unique_samples[x] for x in values_expander]

# We're a bit lazy : we represent bits as a numpy array.
encoded_as_7bits = np.zeros(7 * len(WAV), dtype=bool)
for i in range(len(WAV)):
    encoded_as_7bits[i*7:(i+1)*7] = code_map[WAV[i]]

print(f"Q16: Encoded sound is {len(encoded_as_7bits)//8} bytes long, " +
      f"original was {len(WAV)}")

# Make sure we can decode it
POWERS = np.array([2**(6-i) for i in range(7)])

def decode_7bits(data, decode_map):
    """ Decode a sequence of bits (data) from 7 bits block to 8 bits
    blocks. The conversion from 7 to 8 bits is given by the
    decode_map.
    """
    decoded = np.zeros(len(data) // 7, dtype=np.uint8)
    for j, i in enumerate(range(0, len(data), 7)):
        b = data[i:i+7]
        decoded[j] = decode_map[np.dot(b, POWERS)]
    return decoded

decoded = decode_7bits(encoded_as_7bits, decode_map)
assert (decoded == WAV).all(), "coding/decoding failed in some way"

print(f"Q16: {np.count_nonzero(decoded != WAV)} errors found")

# QUESTION 17


def add_noise(data):
    """Add noise to data, a mutable sequence of bits

    The way we flip bits is not exactly a uniform distribution. But
    it's good enough for the test (on average).
    """
    P = 0.01
    indices = random.sample(range(len(data)), int(len(data)*P))
    for i in indices:
        data[i] = not data[i]


print(f"Q17: Generating noise")
encoded_as_7bits_with_error = encoded_as_7bits.copy()
add_noise(encoded_as_7bits_with_error)

# Simulating the reception (reconstructing the WAV file)
decoded = decode_7bits(encoded_as_7bits_with_error, decode_map)

print(f"Q17: {np.count_nonzero(decoded != WAV)} errors found")

scipy.io.wavfile.write("decoded_with_errors.wav", RATE, decoded)
print("Q17: Wrote decoded_with_errors.wav")

plt.figure()
plt.plot(decoded)
plt.title("Decoded sound, from noisy channel")
plt.xlabel("Samples indices")
plt.ylabel("Samples values")
plt.savefig("q17.pdf")

# QUESTION 18

DATA_BITS = 4
CODE_BITS = 7
hamming = HammingCode(DATA_BITS, CODE_BITS)

# Simplifying code a bit by padding the data.
bits = np.append(encoded_as_7bits, [False] * (DATA_BITS - len(encoded_as_7bits) % DATA_BITS))
assert len(bits) % DATA_BITS == 0, "Padding is necesarry"

print(f"Q18: Encoding {len(bits)} bits")
hamming_bits = bitarray.bitarray()
j = 0
for i in range(0, len(bits), DATA_BITS):
    data = bits[i:i+DATA_BITS].tolist()
    hcode = hamming.encode(data)
    hamming_bits.extend(bitarray.bitarray(list(hcode)))
    j += CODE_BITS
print(f"Q18: Encoded message length is {len(hamming_bits)} bits")

# QUESTION 19

print("Q19: Adding noise")
add_noise(hamming_bits)

print("Q19: Decoding and error-correction")
decoded_bits = bitarray.bitarray()
for i in range(0, len(hamming_bits), CODE_BITS):
    hcode = hamming_bits[i:i+CODE_BITS]
    code = hamming.decode(hcode.tolist())
    decoded_bits.extend(bitarray.bitarray(list(code)))

assert len(bits) == len(decoded_bits),\
    "The decoded data has not the same length as the original data!" + \
    f" {len(decoded_bits)} != {len(bits)}"

error = 0
for i in range(len(bits)):
    if bits[i] != decoded_bits[i]:
        error += 1

print(f"Q19: After Hamming correction, {error} errors left on {len(bits)} bits=> {error/len(bits):.3f}")

decoded = np.zeros(len(WAV), dtype=np.uint8)
for i in range(len(WAV)):
    v = np.dot(decoded_bits[i*7:(i+1)*7].tolist(),
               np.array([64, 32, 16, 8, 4, 2, 1]))
    decoded[i] = decode_map[v]

scipy.io.wavfile.write("decoded_corrected.wav", RATE, decoded)
print("Q19: Wrote decoded_corrected.wav")

plt.figure()
plt.plot(decoded)
plt.title("Decoded sound, with error correction, noisy channel")
plt.xlabel("Samples indices")
plt.ylabel("Samples values")
plt.savefig("q19.pdf")


# QUESTION 20

# Here is a notebook of various computations done while investigating
# question 20.

print()
P = 0.01
def errors(data_bits, code_bits):
    r = code_bits / data_bits
    data_len = len(WAV)*7
    nb_errors = round(P*data_len*r)
    print(f"Q20: Hamming {data_bits}/{code_bits} ")
    print(f"Q20: => coded signal is {math.ceil(data_len*r)} bits long => {nb_errors} errors")
    print(f"Q20: => communication rate {1/r:.2f}")

    return data_len*r, nb_errors

len_4_7, nb_err_4_7 = errors(4,7)
len_11_15, nb_err_11_15 = errors(11,15)

print(f"Q20: {(11/15)/(4/7)}")
print(f"Q20: 4,7 -> 11,15 : Reduction in size by a factor of {(1/len_4_7) * len_11_15:.2f}")
print(f"Q20: 4,7 -> 11,15 : Reduction in errors by a factor of {(1/nb_err_4_7) * nb_err_11_15:.2f}")
