import sys
import math
from io import StringIO
from collections import Counter
from contextlib import redirect_stdout

import numpy as np
import matplotlib.pyplot as plt
try:
    from bitarray import bitarray
except ModuleNotFoundError as ex:
    print("Run pip install bitarray !")
    exit()

from p2_LZ77 import LZ77_encoder, LZ77_decoder, \
    compute_compression_rate_for_LZ77, lz77_cached_compression
from p2_online_lz import online_lz_compress, online_lz_decompress, \
    code_ascii_char, decode_ascii_char, code_binary_char, decode_binary_char
from p2_huffman import build_huffman_tree, build_codebooks, encode, decode, \
    compute_leaves_codes, node_to_neato, decode_one_symbol
from p2_utils import bits_to_represent, compress_counts, decompress_counts


INPUT_FILE = "genome.txt"
CODON_LEN = 3
GENOME_TEXT = "".join(np.genfromtxt(INPUT_FILE, dtype='str'))


def entropy(a):
    s = sum(a)
    h = 0
    for x in a:
        p = x/s
        h += p*math.log2(p)
    return - h

# #ABABAC -> A,B,C ou AB AC
# print(entropy([3,2,1]) / math.log2(3))
# print(entropy([2,1]) / math.log2(2))


""" Q1 Implement a function that returns a binary Huffman code for a
given probability distribution. Give the main steps of your
implementation. Explain how to extend your function to generate a
Huffman code of any alphabet size. Verify your code on Exercise 7 of
the second list of exercises, and report the output of your code for
this example. """

ex7_freq = [0.05, 0.10, 0.15, 0.15, 0.2, 0.35]
symbols = [f"{x:.2f}" for x in ex7_freq]
top_node = build_huffman_tree(dict(zip(symbols, ex7_freq)))
leaves = compute_leaves_codes(top_node)
node_to_neato("graph.dot", top_node, leaves)

""" Q2. Given a sequence of symbols, implement a function that returns
a dictionary and the encoded sequence using the on-line Lempel-Ziv
algorithm (see State of the art in data compression, slide
50/53). Reproduce and report the example given in the course."""

slide_50 = "1 0 11 01 010 00 10".replace(" ", "")
tuples, compressed, prefixes = online_lz_compress(StringIO(slide_50), code_binary_char, tuples_out=True)
assert compressed == "1 00 011 101 1000 0100 0010".replace(" ", "")
assert online_lz_decompress(compressed, decode_binary_char) == slide_50
print()
print(f"Q2: Prefixes dictionary : {prefixes}")
print(f"Q2: (addr, bit) : {tuples}")
print(f"Q2: U: {compressed}")

""" Q4. Implement a function that returns the encoded sequence using the
LZ77 algorithm as described by Algorithm 1 given an input string
and a sliding window size l. Reproduce the example given in Figure
2 with l = 7."""

S_FIGURE_2 = "abracadabrad"
print()
print(f"Q4: Original string : {S_FIGURE_2}")
print(f"Q4: encoded string : {LZ77_encoder(S_FIGURE_2, 7)}")


""" Q5.

A/ Estimate the marginal probability distribution of all codons from
the given genome,

B/ and determine the corresponding binary Huffman code and the encoded
genome.  Give the total length of the encoded genome and the
compression rate. """


def codons_iterator(genome):
    for i in range(0, len(genome), CODON_LEN):
        yield genome[i:i+CODON_LEN]


codons_cnt = Counter(codons_iterator(GENOME_TEXT))
CODONS = sorted(codons_cnt.keys())


# B/ Huffman tree

top_node = build_huffman_tree(codons_cnt)
code_map, decode_map = build_codebooks(top_node)
compressed = encode(codons_iterator(GENOME_TEXT), code_map)
# Validate that compression works by decompressing
assert GENOME_TEXT == "".join(decode(
    compressed, decode_map)), "Decompressed data is not the same as compressed data"

ratio = (len(GENOME_TEXT)*8) / len(compressed)
print()
print(f"Q5: Genome size = {len(GENOME_TEXT)*8} bits; Compressed size = {len(compressed)} bits; ratio={ratio:.2f}")


# A/ Marginal probabilities

marginal_probabilities = dict()
f = sum(codons_cnt.values())
for key, value in codons_cnt.items():
    marginal_probabilities[key] = value/f

with open('Huffman_result.inc', 'w') as f:
    with redirect_stdout(f):
        all_keys = [p[0] for p in
                    sorted([(k, cnt)
                            for k, cnt in marginal_probabilities.items()],
                           key=lambda p: p[1], reverse=True)]
        two_columns = zip(all_keys[:len(all_keys)//2],
                          all_keys[len(all_keys)//2:])

        for c1, c2 in two_columns:
            print(f"{c1} & {marginal_probabilities[c1]:.4f} & {code_map[c1]} & {c2}" +
                  f" & {marginal_probabilities[c2]:.4f} & {code_map[c2]}\\\\")


""" Q6. Give the expected average length for your Huffman
code. Compare this value with (a) the empirical average length, and
(b) theoretical bound(s). Justify.  """


print(f"Entropy of symbols : {entropy(Counter(GENOME_TEXT).values()):.2f}")
print(f"Entropy of codons  : {entropy(codons_cnt.values()):.2f}")

prob = np.array(list(marginal_probabilities.values()), dtype=float)
huffman_codes_lens = np.array([len(code_map[k]) for k in codons_cnt.keys()])
expected_average_length = np.sum(prob*huffman_codes_lens)

print("Q6: expected average length : " +
      f"{expected_average_length:.3f} bits per symbol")

# Here we compute what it would take if we'd store the
# frequencies table so that a decoder got the complete
# information to perform the decompression (we do a rough
# calculation here, it's not bit-exact).

frequencies_length = len(codons_cnt) * bits_to_represent(max(codons_cnt.values()))
total_compressed_length = frequencies_length + len(compressed)

print("Q6: empirical average length : " +
      f"{total_compressed_length} bits / {len(GENOME_TEXT) / CODON_LEN:.1f} symbols = " +
      f"{total_compressed_length/(len(GENOME_TEXT)//CODON_LEN):.3f} bits per symbol")

# Calculate the entropy for the bounds
entropy = - np.sum(prob*np.log2(prob))
print(f"Q6: entropy of source alphabet is : {entropy:.2f}")


""" Q7. Plot the evolution of the empirical average length of the
encoded genome using your Huffman code for increasing input genome
lengths. Discuss your result.  """

if not ("skip7" in sys.argv):
    # Make sure the step is a multiple of codon length
    STEP = ((len(GENOME_TEXT)//100) // CODON_LEN) * CODON_LEN

    # We'll do two graphs. One with fixed Huffman code and
    # one with recomputed Huffman codes. This prepares
    # for the first one.

    const_huff_code_map, _ = build_codebooks(
        build_huffman_tree(
            Counter(
                codons_iterator(GENOME_TEXT))))

    x_axis = []
    empirical_avg_lens = []
    empirical_avg_lens_const_huffman = []

    for i in range(STEP, len(GENOME_TEXT), STEP):
        g = GENOME_TEXT[0:i]
        codons_in_subgenome = len(g) // CODON_LEN
        x_axis.append(round(100*len(g) / len(GENOME_TEXT)))

        # First graph : one Huffman per sub-genome

        codons_cnt = Counter(codons_iterator(g))
        top_node = build_huffman_tree(codons_cnt)
        code_map, decode_map = build_codebooks(top_node)
        compressed_bits = encode(codons_iterator(g), code_map)
        assert g == "".join(decode(compressed_bits, decode_map)), "Compression went wrong"

        empirical_avg_lens.append(len(compressed_bits) / codons_in_subgenome)

        # Second graph : one Huffman for all sub-genomes

        compressed_bits = encode(codons_iterator(g), const_huff_code_map)
        empirical_avg_lens_const_huffman.append(len(compressed_bits) / codons_in_subgenome)

        print(f"{x_axis[-1]} {empirical_avg_lens[-1]:.3f} {empirical_avg_lens_const_huffman[-1]:.3f}")

    plt.figure()
    plt.plot(x_axis, empirical_avg_lens, label="One Huffman per subgenome")
    plt.plot(x_axis, empirical_avg_lens_const_huffman,
             label="One Huffman for all")
    plt.title("Empirical average length")
    plt.xlabel("Data size (% of the total genome size)")
    plt.ylabel("Bits per codon")
    plt.legend()
    plt.savefig("q7.pdf")
    plt.show()

    print("Q7: Empirical Average lengths : ", empirical_avg_lens)


""" Q9. Encode the genome using the on-line Lempel-Ziv algorithm. Give
the total length of the encoded genome and the compression rate."""

coded_bin = online_lz_compress(StringIO(GENOME_TEXT), code_ascii_char)
decoded = online_lz_decompress(coded_bin, decode_ascii_char)
assert decoded == GENOME_TEXT, "something went wrong in the compression or decompression"

print()
print("Q9: encode genome with online LZ")
print("Q9: total length of source genome, without spaces : " +
      f"{len(GENOME_TEXT)} symbols, {len(GENOME_TEXT)*8} bits")
print(f"Q9: total length of encoded genome : {len(coded_bin)} bits")
print("Q9: compression rate (lecture 4, slide 18) : " +
      f"{len(GENOME_TEXT*8)} bits / {len(coded_bin)} bits = " +
      f"{len(GENOME_TEXT*8)/len(coded_bin):.2f}.")


""" Q10. Encode the genome using the LZ77 algorithm. Give the total
length of the encoded genome and the compression rate."""

WIN_SIZE = 512*2
tuples = lz77_cached_compression(WIN_SIZE, GENOME_TEXT)

dl_bits = math.ceil(math.log2(WIN_SIZE))
char_bits = 8
tuple_bits = char_bits+2*dl_bits
print()
print(f"Q10: total length of source genome, without spaces : {len(GENOME_TEXT)} symbols, {len(GENOME_TEXT)*8} bits")
print(f"Q10: sliding window size = {WIN_SIZE} => {dl_bits} bits for d and l each")
print(f"Q10: {char_bits} bits per char => {tuple_bits} bits per tuples")
compressed_size, compression_rate = compute_compression_rate_for_LZ77(tuples, WIN_SIZE, GENOME_TEXT)
print(f"Q10: total length of encoded genome : {len(tuples)} tuples * {tuple_bits} bits = {compressed_size} bits")
print(f"Q10: compression rate : {len(GENOME_TEXT)*8} bits / {compressed_size} bits = {len(GENOME_TEXT)*8/compressed_size:.2f} ")

""" 11. Famous data compression algorithms combine the LZ77 algorithm
and the Huffman algorithm.  Explain how these algorithms can be
combined and discuss the interest of the possible combinations.  """

small_c = [c for d, l, c in tuples]
small_l = [l for d, l, c in tuples]
small_d = [d for d, l, c in tuples]

# FIXME Try this
#small_c, small_l, small_d = zip(*tuples)

# --------------------------------------------------------------------
# (l,d,c) -> tuple
tuples_count = Counter(tuples)
plt.plot(list(sorted(tuples_count.values())))

print(f"\n\n2. build_huffman_tree on {len(tuples)} tuples (of which {len(tuples_count)} are unique)")

tree_size = len(tuples_count) * (bits_to_represent(len(small_c)) + bits_to_represent(max(small_d)+1) + bits_to_represent(max(small_l)+1) + bits_to_represent(len(tuples_count)))
print(f"Tree size = {tree_size} bits => {tree_size//8} bytes")


top_node = build_huffman_tree(tuples_count)
code_map, decode_map = build_codebooks(top_node)
compressed_size = sum([len(code_map[t])*cnt for t, cnt in tuples_count.items()]) + tree_size

# --------------------------------------------------------------------
# (l,d,c) -> (l,c) + (d)

dist_count = Counter(small_d)
len_count = Counter(small_l)
char_count = Counter(small_c)
print(char_count)
# char_count = dict(zip(CODONS, [c[codon] for codon in CODONS]))

plt.figure()
plt.plot(list(dist_count.values()))
plt.xlabel("Distances")
plt.ylabel("# occurences")
# plt.show()

plt.figure()
plt.plot(list(len_count.values()))
plt.xlabel("Lengths")
plt.ylabel("# occurences")
# plt.show()

top_node = build_huffman_tree(dist_count)
dist_code_map, dist_decode_map = build_codebooks(top_node)

dist_tree_size = len(dist_count) * (bits_to_represent(max(dist_count.values())+1) + bits_to_represent(max(dist_count.keys())+1))
len_tree_size = len(len_count) * (bits_to_represent(max(len_count.values())+1) + bits_to_represent(max(len_count.keys())+1))
char_tree_size = len(char_count) * bits_to_represent(max(char_count.values())+1)

all_trees_size = dist_tree_size + len_tree_size + char_tree_size

print(f"All tree size = {all_trees_size} bits => {all_trees_size//8} bytes")
top_node = build_huffman_tree(len_count)
len_code_map, len_decode_map = build_codebooks(top_node)

top_node = build_huffman_tree(char_count)
char_code_map, char_decode_map = build_codebooks(top_node)

print(f"Dist counts : {len(dist_count)}")
print(f"Len counts : {len(len_count)}")
print(f"C counts : {len(char_count)}")

print(f"1. l + c + d : build_huffman_tree on {len(tuples)} tuples (of which {len(char_count)} are unique)")

compressed_size = sum([len(char_code_map[c]) + len(len_code_map[l]) + len(dist_code_map[d]) for d,l,c in tuples]) + all_trees_size
plt.figure()
plt.plot(list(sorted(char_count.values())))
plt.show()


# --------------------------------------------------------------------

"""Q12. Encode the genome using the best (according to your answer in
the previous question) combination of LZ77 and Huffman
algorithms. Give the total length of the encoded genome and the
compression rate."""

def lz_with_huffman_encode(sliding_window_size, genome):

    tuples = lz77_cached_compression(sliding_window_size, genome)

    distances, lengths, chars = zip(*tuples)

    dist_count = Counter(distances)
    len_count = Counter(lengths)
    char_count = Counter(chars)

    bits = bitarray()
    # print("Counts for distances")
    bits.extend(compress_counts(dist_count))
    # print("Counts for lengths")
    bits.extend(compress_counts(len_count))
    # print("Counts for chars")
    bits.extend(compress_counts(char_count))

    # Building Huffman trees and codebooks
    top_node = build_huffman_tree(dist_count)
    dist_code_map, _ = build_codebooks(top_node)
    top_node = build_huffman_tree(len_count)
    len_code_map, _ = build_codebooks(top_node)
    top_node = build_huffman_tree(char_count)
    char_code_map, _ = build_codebooks(top_node)

    for d, l, c in tuples:
        # Compress a tuple (d,l,c) into its huffman representation
        # (d -> 0101..., l -> 11000, c->101010), using a different
        # codebook for d,l and c.
        bits.extend(bitarray(dist_code_map[d]))
        bits.extend(bitarray(len_code_map[l]))
        bits.extend(bitarray(char_code_map[c]))

    return bits


def lz_with_huffman_decode(bits):

    total_read_bits = 0
    read_bits, dist_count = decompress_counts(bits)
    total_read_bits += read_bits
    read_bits, len_count = decompress_counts(bits[total_read_bits:])
    total_read_bits += read_bits
    read_bits, char_count = decompress_counts(bits[total_read_bits:], as_type='char')
    total_read_bits += read_bits

    # assert sum(dist_count.values()) == sum(len_count.values()) == \
    #     sum(char_count.values()) == len(tuples), "Counts deomcpression went bad"

    top_node = build_huffman_tree(dist_count)
    dist_code_map, dist_decode_map = build_codebooks(top_node)
    top_node = build_huffman_tree(len_count)
    len_code_map, len_decode_map = build_codebooks(top_node)
    top_node = build_huffman_tree(char_count)
    char_code_map, char_decode_map = build_codebooks(top_node)

    dtuples = []
    for i in range(sum(dist_count.values())):
        # total_read_bits+100 : make sure we don't extract all the remaining
        # of bit array to decode_one_symbol function each time => it's a
        # speed optimisation. We can do it because we assume Huffman codes
        # won't be more than 100 bits. Ideally we could compute that number
        # based on Huffman trees themselves.

        read_bits, d = decode_one_symbol(
            bits[total_read_bits:total_read_bits+100], dist_decode_map)
        total_read_bits += read_bits
        read_bits, l = decode_one_symbol(
            bits[total_read_bits:total_read_bits+100], len_decode_map)
        total_read_bits += read_bits
        read_bits, c = decode_one_symbol(
            bits[total_read_bits:total_read_bits+100], char_decode_map)
        total_read_bits += read_bits
        dtuples.append((d, l, c))

        if len(dtuples) % 10000 == 0:
            print(len(dtuples))

    print(f"Bits read = {total_read_bits}; decompresseed tuples = {len(dtuples)}")

    # assert len(tuples) == len(dtuples)
    # for i in range(len(tuples)):
    #     assert dtuples[i] == tuples[i], \
    #         f"Decompression failed on tuples {tuples[i]} != {dtuples[i]}"

    return LZ77_decoder(dtuples)


bits = lz_with_huffman_encode(WIN_SIZE, GENOME_TEXT)
print()
print("Q12: total length of source genome, without spaces : " +
      f"{len(GENOME_TEXT)} symbols, {len(GENOME_TEXT)*8} bits on disk")
print(f"Q12: LZ77+Huffman coded file length : {len(bits)} bits")

rate = len(GENOME_TEXT)*8 / len(bits)
print(f"Q12: Compression rate : {len(GENOME_TEXT)*8} bits / " +
      f"{len(bits)} bits = {rate:.2f}")

assert "".join(lz_with_huffman_decode(bits)) == GENOME_TEXT, "Decompression didn't work"


""" Q13. Report the total lengths and compression rates using (a) LZ77
and (b) the combination of LZ77 and Huffman, to encode the genome for
different values of the sliding window size l. Compare your result
with the total length and compression rate obtained using the on-line
Lempel-Ziv algorithm.  Discuss your results. """

with open("q13.inc", "w") as output:
    for sliding_window_size in [256, 512, 1024, 2048, 4096, 8192, 16384,
                                32768, 65536, 2**17, 2**18]:
        # LZ77 only
        tuples = lz77_cached_compression(sliding_window_size, GENOME_TEXT)
        compressed_size, compression_rate = compute_compression_rate_for_LZ77(
            tuples, sliding_window_size, GENOME_TEXT)

        # LZ77 + Huffman
        bits = lz_with_huffman_encode(sliding_window_size, GENOME_TEXT)
        lz77_huffman_rate = len(GENOME_TEXT)*8 / len(bits)

        txt = f"{sliding_window_size} & {compressed_size} & " + \
            f"{compression_rate:.2f} & {len(bits)} & " + \
            f"{lz77_huffman_rate:.2f} \\\\"
        print(txt)
        output.write(txt + "\n")
