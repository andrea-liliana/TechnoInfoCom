import math
import os.path
import pickle
from datetime import datetime


def LZ77_encoder(input_text, SWSIZE):
    """ Return a list of (distance, length, character) tuples.
    """

    def peek(ndx):
        if ndx < 0:
            # We assume None is never present in the input text
            return None
        else:
            return input_text[ndx]

    compressed = []
    i = 0
    while i < len(input_text):

        # We'll look for the longest match in the sliding window

        longest_prefix_pos = 0
        longest_prefix_len = 0

        # For that, we go over all possible prefix starts, in the
        # sliding window located right before the current character.

        # This range is easy to understand (we just run the sliding
        # windows from left to right) : r = range(i - SWSIZE, i). But
        # the one we use is trickier, it's because in the TA's example
        # he goes the other way around (right to left)...

        for pfx_start in range(i-1, i-SWSIZE-1, -1):

            pfx_len = 0
            prefix = ""

            # The first condition is here to avoid going past EOF.
            # The second is for extending the repeated part.
            # Note that the repeated string can go past the sliding
            # window.

            while i+pfx_len < len(input_text) - 1 and\
                  peek(pfx_start + pfx_len) == peek(i+pfx_len):

                prefix += peek(pfx_start+pfx_len)
                pfx_len += 1

                # assert pfx_start+pfx_len < len(input_text)
                # assert i+pfx_len < len(input_text)

            # Is this prefix better ?
            if pfx_len > longest_prefix_len:
                # print(f"Best {pfx_start}")
                longest_prefix_pos = pfx_start
                longest_prefix_len = pfx_len

        # We take prefix of length 2 or more else they are
        # just a regular single symbol and not worht a repetition.
        if longest_prefix_len > 1:
            d, l, c = i - longest_prefix_pos, longest_prefix_len, input_text[i + longest_prefix_len]
        else:
            d, l, c = 0, 0, input_text[i]

        compressed.append((d, l, c))
        i += l + 1

    return compressed


def compute_compression_rate_for_LZ77(tuples, sliding_window_size, genome):
    dl_bits = math.ceil(math.log2(sliding_window_size))
    char_bits = 8
    tuple_bits = char_bits+2*dl_bits
    compressed_size_in_bits = len(tuples)*tuple_bits
    compression_rate = len(genome)*8/compressed_size_in_bits
    return compressed_size_in_bits, compression_rate


# The following code is to avoid recompressing the genome
# each time we run the program.

def lz77_cached_compression(sliding_window_size, genome):
    cache_name = f"LZ77Cache{sliding_window_size}.dat"
    if not os.path.exists(cache_name):
        print(f"Crunching with LZ77, sliding window {sliding_window_size}. " +
              "This can take from 2 minutes from 5 hours depending on " +
              "sliding window size.")
        chrono = datetime.now()
        tuples = LZ77_encoder(genome, sliding_window_size)
        print(f"Compression took {datetime.now() - chrono}")
        assert "".join(LZ77_decoder(tuples)) == genome, \
            "LZ77 compression went wrong"
        with open(cache_name, "wb") as fout:
            pickle.dump(tuples, fout)
    else:
        with open(cache_name, "rb") as fin:
            tuples = pickle.load(fin)

    return tuples


def LZ77_decoder(encoded):
    decoded = []
    for d, l, c in encoded:
        if l > 0:
            ofs = len(decoded) - d

            # This loop allows symbol repetitions
            # to be defined past the end of the
            # sliding window.

            for i in range(l):
                decoded.append(decoded[ofs+i])

        decoded.append(c)

    return decoded


if __name__ == "__main__":
    """ Q4. Implement a function that returns the encoded sequence using the
    LZ77 algorithm as described by Algorithm 1 given an input string
    and a sliding window size l. Reproduce the example given in Figure
    2 with l = 7."""

    S = "abracadabrad"
    print(S)
    print(LZ77_encoder(S, 7))
