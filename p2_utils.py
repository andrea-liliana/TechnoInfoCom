import math
from bitarray import bitarray


def bits_to_represent(nb_values):
    # Discontinuous ! f == 0 in x==1 !
    assert nb_values >= 1
    return math.ceil(math.log2(nb_values))


def int32_to_bits(i):
    # We take 32 bits to be safe.
    return bitarray(f"{i:032b}")


def compress_values(values):
    if type(values[0]) == int:
        # +1 is tricky ! We assume we encode values 0 to max !
        bits_per_value = bits_to_represent(max(values)+1)
        fmt = lambda x: f"{x:0{bits_per_value}b}"
    elif type(values[0]) == str and len(values[0]) == 1:
        bits_per_value = 8
        fmt = lambda x: f"{ord(x):0{bits_per_value}b}"
    else:
        raise Exception("Only int's and one-char-strings are supported")

    # print(f"Compress: bits_per_value={bits_per_value}, " +
    #       f"nb_values={len(values)} [{min(values)}, {max(values)}]")

    bits = bitarray()
    bits.extend(int32_to_bits(bits_per_value))
    bits.extend(int32_to_bits(len(values)))

    for v in values:
        bits.extend(bitarray(fmt(v)))
    return bits


def decompress_values(bits, as_type='int'):
    bits_per_value = int(bits[0:32].to01(), 2)
    nb_values = int(bits[32:64].to01(), 2)

    cut_bits = [bits[i:i+bits_per_value].to01() for i in range(64,64+nb_values*bits_per_value, bits_per_value)]

    if as_type == 'int':
        v = [int(x, 2) for x in cut_bits]
    elif as_type == "char":
        v = [chr(int(x, 2)) for x in cut_bits]
    else:
        raise Exception("Only int's and one-char-strings are supported")

    # print(f"Decompress: bits_per_value={bits_per_value}, nb_values={nb_values}")
    # Return bits read, decompressed counts dictionary
    return 64 + bits_per_value*nb_values, v


def compress_counts(counts):
    # for v in counts.keys():
    #     assert v > 0, "0 is reserved for end of stream symbol"

    bits = bitarray()
    bits.extend(compress_values(list(counts.keys())))
    bits.extend(compress_values(list(counts.values())))
    return bits


def decompress_counts(bits, as_type='int'):
    total_read_bits = 0
    read_bits, k = decompress_values(bits[total_read_bits:], as_type)
    total_read_bits += read_bits
    read_bits, v = decompress_values(bits[total_read_bits:])
    total_read_bits += read_bits

    # Return bits read, decompressed counts dictionary
    return total_read_bits, dict(zip(k, v))


if __name__ == "__main__":

    assert bits_to_represent(1) == 0
    assert bits_to_represent(255) == 8
    assert bits_to_represent(256) == 8
    assert bits_to_represent(257) == 9

    # Test compress values
    bits = compress_values([1, 2, 5, 6, 7, 8, 9, 10, 100, 101, 102, 103, 256])
    decomp_bits, values = decompress_values(bits)
    assert decomp_bits == len(bits)

    # Test compress counts
    bits = compress_counts({1: 2, 10: 11, 12: 256})
    decomp_bits, cmap = decompress_counts(bits)
    assert decomp_bits == len(bits), f"{decomp_bits} != {len(bits)}"
