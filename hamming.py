import numpy as np


def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


DATA_BITS, CODE_BITS = 4, 7
PARITY_BITS = CODE_BITS - DATA_BITS


parities = np.invert(np.eye(PARITY_BITS, dtype=np.bool))
one_row = np.ones((1, PARITY_BITS))
parities = np.append(parities, one_row, axis=0)

G = np.append(np.eye(DATA_BITS, dtype=np.bool), parities, axis=1)
print(G)

H = np.append(np.transpose(parities),
              np.eye(PARITY_BITS, dtype=np.bool), axis=1)
print("H")
print(H)

x = np.array([1, 1, 1, 1])

correct_code = (x @ G) % 2
print()

# Introducing an error
code = correct_code.copy()
code[0] = 0
#code[1] = 0

parity_check = (H @ code) % 2
print(f"Correct code:{correct_code} Received code:{code} Parity:{parity_check}")


POWER2 = np.array([2**i for i in range(PARITY_BITS)])

correction_map = dict()
mle_map = dict()

for i in range(2**DATA_BITS):
    # See https://en.wikipedia.org/wiki/Decoding_methods#Syndrome_decoding
    error = bin_array(i, DATA_BITS)
    h_e = (np.transpose(parities) @ error) % 2
    print(f"{error} -> {h_e}")

    ndx = np.dot(POWER2,h_e)
    if ndx not in mle_map or np.sum(error) < mle_map[ndx]:
        mle_map[ndx] = np.sum(error)
        correction_map[ndx] = error

print(correction_map[np.dot(POWER2,parity_check)])
