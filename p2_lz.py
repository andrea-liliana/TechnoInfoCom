import os
import sys
import math
import heapq
from io import StringIO
from collections import Counter
import os.path
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from bitarray import bitarray

from p2_LZ77 import LZ77_encoder, LZ77_decoder

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




class Node:
    def __init__(self, left_child=None, right_child=None, weight=None, symbol=None):
        self.left_child = left_child
        self.right_child = right_child

        if self.has_both_children():
            assert weight is None and symbol is None
            self.weight = self.left_child.weight + self.right_child.weight
            self.symbol = None
        else:
            assert weight > 0 and symbol is not None, f"Weight={weight}, symbol={symbol}"
            self.weight = weight
            self.symbol = symbol

        assert (left_child is None and right_child is None) or self.has_both_children()
        self.code = None

    def has_both_children(self):
        return self.left_child is not None and self.right_child is not None

    def __eq__(self, other):
        return self.weight == other.weight

    def __lt__(self, other):
        return self.weight < other.weight


def build_huffman_tree(symbols_cnts: dict):
    # Create leaves of the tree
    nodes = []
    for symbol, cnt in symbols_cnts.items():
        nodes.append((cnt, Node(None, None, cnt, symbol)))

    # Order leaves by weights, heapq is a min-heap
    heapq.heapify(nodes)

    # Build the tree bottom up
    while len(nodes) > 1:
        # Pop the two nodes with the lowest weights
        left = heapq.heappop(nodes)[1]
        right = heapq.heappop(nodes)[1]

        new_node = Node(left, right)
        heapq.heappush(nodes, (new_node.weight, new_node))

    # return the remainging node which is the top node
    # of the tree
    return nodes[0][1]


def compute_leaves_codes(node: Node, prefix=""):
    if node.has_both_children():
        a = compute_leaves_codes(node.left_child, prefix + "0")
        b = compute_leaves_codes(node.right_child, prefix + "1")
        return a+b
    else:
        assert node.left_child is None and node.right_child is None
        node.code = prefix
        return [node]

def gid(node):
    return f"{int(id(node))}"

def draw_neato_tree(fout, node):

    if node.has_both_children():
        fout.write(f"{gid(node)} [label=\"{node.weight:.2f}\"]\n")

        fout.write(f"{gid(node)} -- {gid(node.left_child)} [label=\"0\"]\n")
        draw_neato_tree(fout, node.left_child)
        fout.write(f"{gid(node)} -- {gid(node.right_child)} [label=\"1\"]\n")
        draw_neato_tree(fout, node.right_child)
    else:
        fout.write(f"{gid(node)} [label=\"{node.symbol}\\n{node.code}\"]\n")

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

with open("graph.dot", "w") as fout:
    fout.write("graph HuffmanTree {\n")
    draw_neato_tree(fout, top_node)
    r = " ".join([gid(n)+";" for n in leaves])
    fout.write(f"{{rank = same; {r}}}\n")
    fout.write("}\n")



def bits_to_represent(nb_values):
    # Discontinuous ! f == 0 in x==1 !
    assert nb_values >= 1
    return math.ceil(math.log2(nb_values))

def code_binary_char(c):
    return c

def code_ascii_char(c):
    return f"{ord(c):08b}"

def online_lz_compress(data, code_char, tuples_out=False):

    # The whole point of this trickery is to detect
    # EOF one char ahead. This way we can tell when
    # we're processing the last character of the
    # input file

    EOF_MARK = ''
    c = data.read(1)  # Read one byte
    assert c != EOF_MARK, "Compressing empty file is not supported"
    next_char = data.read(1)
    eof = next_char == EOF_MARK

    coded_bin = StringIO()
    prefixes = {"": 0}
    prefix = ""
    tuples = ""

    while True:
        if not eof and prefix + c in prefixes:
            prefix = prefix + c
        else:
            # We have :
            # - either EOF on next char,
            # - either prefix+c not in prefixes.
            # But in both cases we store the (prefix,c). That's
            # especially useful when EOF is met since it
            # allows us to terminate the output stream cleanly.

            # Number of bits necessary to represent the length
            # of the prefixes table.
            # FIXME this is not really fast.
            l = bits_to_represent(len(prefixes))

            # Append (prefix's number, c) to the output stream.
            char = code_char(c)

            #char = f"{ord(c):08b}"
            if l == 0:
                pfx = ""
            else:
                pfx = f"{np.binary_repr(prefixes[prefix],l)}"
            coded_bin.write(f"{pfx}{char}")

            if tuples_out:
                if l == 0:
                    pfx = ""
                else:
                    pfx = f"{np.binary_repr(prefixes[prefix],l)}"
                tuples += f"({pfx},{char}) "

            # Record the new prefix and give it a number
            prefixes[prefix + c] = len(prefixes)

            # Prepare for next iteration
            prefix = ""

        if next_char != EOF_MARK:
            c = next_char
            next_char = data.read(1)
            eof = next_char == EOF_MARK
        else:
            break

    if not tuples_out:
        return coded_bin.getvalue()
    else:
        return tuples, coded_bin.getvalue()


def decode_binary_char(data, ndx):
    return data[ndx], 1

def decode_ascii_char(data, ndx):
    return chr(int(data[ndx:ndx+8], 2)), 8

def online_lz_decompress(coded_bin, decode_char):
    ndx = 0
    decoded = StringIO()
    prefixes = {0: ""}
    while ndx < len(coded_bin):

        l = bits_to_represent(len(prefixes))
        if l > 0:
            prefix_code = int(coded_bin[ndx:ndx+l], 2)
        else:
            prefix_code = 0

        char, skips = decode_char(coded_bin, ndx+l)

        decoded.write(prefixes[prefix_code] + char)
        prefixes[len(prefixes)] = prefixes[prefix_code] + char
        #ndx = ndx+l+8
        ndx = ndx+l+skips

    return decoded.getvalue()



def int32_to_bits(i):
    # We take 32 bits to be safe.
    return bitarray(f"{i:032b}")

def compress_values(values):
    if type(values[0]) == int:
        bits_per_value = bits_to_represent(max(values)+1) # +1 is tricky ! We assume we encode values 0 to max ! FIXME !
        fmt = lambda x: f"{x:0{bits_per_value}b}"
    elif type(values[0]) == str and len(values[0]) == 1:
        bits_per_value = 8
        fmt = lambda x: f"{ord(x):0{bits_per_value}b}"
    else:
        raise Exception("Only int's and one-char-strings are supported")

    # print(f"Compress: bits_per_value={bits_per_value}, nb_values={len(values)} [{min(values)}, {max(values)}]")

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
    return total_read_bits, dict(zip(k,v))

# Test compress values
bits = compress_values([1,2,3,4,5,6,7,8,9,10,100,101,102,103,256])
decomp_bits, values = decompress_values(bits)
assert decomp_bits == len(bits)

# Test compress counts
bits = compress_counts({1:2, 10:11, 12:256})
decomp_bits, cmap = decompress_counts(bits)
assert decomp_bits == len(bits), f"{decomp_bits} != {len(bits)}"


""" Q2. Given a sequence of symbols, implement a function that returns
a dictionary and the encoded sequence using the on-line Lempel-Ziv
algorithm (see State of the art in data compression, slide
50/53). Reproduce and report the example given in the course."""

slide_50 = "1 0 11 01 010 00 10".replace(" ", "")
tuples, compressed = online_lz_compress(StringIO(slide_50), code_binary_char, tuples_out=True)
assert compressed == "1 00 011 101 1000 0100 0010".replace(" ", "")
assert online_lz_decompress(compressed, decode_binary_char) == slide_50
print()
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


def build_codebooks(top_node):
    # Affect a code to each leaf node
    d = compute_leaves_codes(top_node, "")

    # Build maps from/to symbol to/from Huffman codes
    code_map = dict()
    decode_map = dict()
    for node in sorted(d, key=lambda n: n.weight):
        #print(f"{node.symbol} {node.weight:5d} {node.code}")
        code_map[node.symbol] = node.code
        decode_map[node.code] = node.symbol

    return code_map, decode_map


def codons_iterator(genome):
    for i in range(0, len(genome), CODON_LEN):
        yield genome[i:i+CODON_LEN]


def encode(symbol_iter, code_map):
    """ Convert a serie of symbols into a serie of
    corresponding codewords (expected to be string
    representation of binary codes, eg 001100).

    - symbol_iter : an iterator which will give all
    the symbols of the data to compress, on by one,
    in order.
    - code_map : map from symbol to codeword.

    Note that data end detection rely on the iterator end
    (here it's detected by Python). So we don't add an
    additionaly symbol to represent the end of file.
    """

    file_str = StringIO()
    for symbol in symbol_iter:
        file_str.write(code_map[symbol])

    return file_str.getvalue()


def decode_one_symbol(compressed, decode_map):
    prefix = ""
    for c in compressed:
        assert c in (True, False), f"Unexpected char : {c}"
        if c:
            prefix += "1"
        else:
            prefix += "0"

        if prefix in decode_map:
            return len(prefix), decode_map[prefix]

    raise Exception("EOF unexpected")

def decode(compressed, decode_map, nb_symbols = 2**31):
    # File end is detected by file size. See remark in
    # encode() funtcion.

    ns = 0
    prefix = ""
    file_str = StringIO()
    for c in compressed:
        prefix += c

        if prefix in decode_map:
            file_str.write(decode_map[prefix])
            prefix = ""

            if ns < nb_symbols-1:
                ns += 1
            else:
                break

    return file_str.getvalue()


""" Q5.

A/ Estimate the marginal probability distribution of all codons from
the given genome,

B/ and determine the corresponding binary Huffman code and the encoded
genome.  Give the total length of the encoded genome and the
compression rate. """

codons_cnt = Counter(codons_iterator(GENOME_TEXT))
CODONS = sorted(codons_cnt.keys())

print(entropy(Counter(GENOME_TEXT).values()))
print(entropy(codons_cnt.values()))

top_node = build_huffman_tree(codons_cnt)
code_map, decode_map = build_codebooks(top_node)
compressed = encode(codons_iterator(GENOME_TEXT), code_map)
# Validate that compression works by decompressing
assert GENOME_TEXT == "".join(decode(
    compressed, decode_map)), "Decompressed data is not the same as compressed data"

ratio = (len(GENOME_TEXT)*8) / len(compressed)
print(f"Q5: Genome size = {len(GENOME_TEXT)*8} bits; Compressed size = {len(compressed)} bits; ratio={ratio:.2f}")


""" Q6. Give the expected average length for your Huffman
code. Compare this value with (a) the empirical average length, and
(b) theoretical bound(s). Justify.  """

# Calculate probabilities
f = sum(codons_cnt.values())
for key, value in codons_cnt.items():
    codons_cnt[key] = round(value/f,5)

prob = []
for value in codons_cnt.values():
    prob.append(value)

prob = np.array(prob)
#print(prob)
prob = prob.astype(float)

# Calculate the lengths of the Huffman codes
# l = []
# for value in code_map.values():
#   l.append(len(value))
# l = np.array(l)

l = np.array([len(value) for value in code_map.values()])

expected_average_length = np.sum(prob*l)

# Get table q5
from contextlib import redirect_stdout
with open('out.txt', 'w') as f:
    with redirect_stdout(f):
        for k in codons_cnt.keys() & code_map.keys():
            print(k,"&", codons_cnt[k],"&", code_map[k] , "\\")

print(f"Q6: expected_average_length : {expected_average_length:.2f} bits")


print(f"Q6: empirical average length : {len(compressed)} bits / {len(GENOME_TEXT)} symbols = {len(compressed)/(len(GENOME_TEXT)//CODON_LEN):.2f}")

# Calculate the entropy for the bounds
entropy = - np.sum(prob*np.log2(prob))

print(f"Q6: entropy of source alphabet is : {entropy:.2f}")

exit()

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
    plt.plot(x_axis, empirical_avg_lens_const_huffman, label="One Huffman for all")
    plt.title("Empirical average length")
    plt.xlabel("Data size (% of the total genome size)")
    plt.ylabel("Bits per codon")
    plt.legend()
    plt.savefig("q7.pdf")
    plt.show()

print("q7", empirical_avg_lens)



""" Q9. Encode the genome using the on-line Lempel-Ziv algorithm. Give
the total length of the encoded genome and the compression rate."""

coded_bin = online_lz_compress(StringIO(GENOME_TEXT), code_ascii_char)
decoded = online_lz_decompress(coded_bin, decode_ascii_char)
assert decoded == GENOME_TEXT, "something went wrong in the compression or decompression"

print()
print("Q9: encode genome with online LZ")
print(f"Q9: total length of source genome, without spaces : {len(GENOME_TEXT)} symbols, {len(GENOME_TEXT)*8} bits")
print(f"Q9: total length of encoded genome : {len(coded_bin)} bits")
print(f"Q9: compression rate (lecture 4, slide 18) : {len(GENOME_TEXT*8)} bits / {len(coded_bin)} bits = {len(GENOME_TEXT*8)/len(coded_bin):.2f}.")





# The following code is to avoid recompressing the genome
# each time we run the program.

def lz77_cached_compression(sliding_window_size, genome):
    cache_name=f"LZ77Cache{sliding_window_size}.dat"
    if not os.path.exists(cache_name):
        print(f"Crunching with LZ77, sliding window {sliding_window_size}")
        chrono = datetime.now()
        tuples = LZ77_encoder(genome, sliding_window_size)
        print(f"Compression took {datetime.now() - chrono}")
        assert "".join(LZ77_decoder(tuples)) == genome, "LZ77 compression went wrong"
        with open(cache_name,"wb") as fout:
            pickle.dump(tuples, fout)
    else:
        with open(cache_name,"rb") as fin:
            tuples = pickle.load(fin)

    return tuples

def compute_compression_rate_for_LZ77(tuples, sliding_window_size, genome):
    dl_bits = math.ceil(math.log2(sliding_window_size))
    char_bits = 8
    tuple_bits = char_bits+dl_bits
    compressed_size = len(tuples)*tuple_bits
    compression_rate = len(genome)*8/compressed_size
    return compressed_size, compression_rate


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



"""Q12. Encode the genome using the best (according to your answer in
the previous question) combination of LZ77 and Huffman
algorithms. Give the total length of the encoded genome and the
compression rate."""


small_c = [c for d,l,c in tuples]
small_l = [l for d,l,c in tuples]
small_d = [d for d,l,c in tuples]

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

dist_count = Counter(small_d) # 1 -> 001 ,2->110,3->1110,2,5,1,1,1,1,1,4,6,...,1000 -> dict( symbol -> count)
len_count = Counter(small_l)
char_count = Counter(small_c)
print(char_count)
#char_count = dict(zip(CODONS, [c[codon] for codon in CODONS]))

plt.figure()
plt.plot(list(dist_count.values()))
plt.xlabel("Distances")
plt.ylabel("# occurences")
#plt.show()

plt.figure()
plt.plot(list(len_count.values()))
plt.xlabel("Lengths")
plt.ylabel("# occurences")
#plt.show()

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

# Compression ************************************************

def lz_with_huffman_encode(sliding_window_size, genome):

    tuples = lz77_cached_compression(sliding_window_size, genome)

    dist_count = Counter([d for d,l,c in tuples])
    len_count = Counter([l for d,l,c in tuples])
    char_count = Counter([c for d,l,c in tuples])

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

    for d,l,c in tuples:
        # Compress a tuple (d,l,c) into its huffman representation
        # (d -> 0101..., l -> 11000, c->101010), using a different
        # codebook for d,l and c.
        bits.extend(bitarray(dist_code_map[d]))
        bits.extend(bitarray(len_code_map[l]))
        bits.extend(bitarray(char_code_map[c]))

    return bits

bits = lz_with_huffman_encode(WIN_SIZE, GENOME_TEXT)
print()
print(f"Q12: total length of source genome, without spaces : {len(GENOME_TEXT)} symbols, {len(GENOME_TEXT)*8} bits on disk")
print(f"Q12: LZ77+Huffman coded file length : {len(bits)} bits")

# rate = (len(genome)/len(bits)) * (math.log2(4)/math.log2(2))
# print(f"Q12: Compression rate : ({len(genome)} chars /{len(bits)} bits) * ({math.log2(4)} letters / {math.log2(2)} bits) = {rate:.2f}")

rate = len(GENOME_TEXT)*8 / len(bits)
print(f"Q12: Compression rate : {len(GENOME_TEXT)*8} bits / {len(bits)} bits = {rate:.2f}")

# Decompression ************************************************

total_read_bits = 0
read_bits, dist_count = decompress_counts(bits)
total_read_bits += read_bits
read_bits, len_count = decompress_counts(bits[total_read_bits:])
total_read_bits += read_bits
read_bits, char_count = decompress_counts(bits[total_read_bits:], as_type='char')
total_read_bits += read_bits

assert sum(dist_count.values()) == sum(len_count.values()) == sum(char_count.values()) == len(tuples), "Counts deomcpression went bad"

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

    read_bits, d = decode_one_symbol(bits[total_read_bits:total_read_bits+100], dist_decode_map)
    total_read_bits += read_bits
    read_bits, l = decode_one_symbol(bits[total_read_bits:total_read_bits+100], len_decode_map)
    total_read_bits += read_bits
    read_bits, c = decode_one_symbol(bits[total_read_bits:total_read_bits+100], char_decode_map)
    total_read_bits += read_bits
    dtuples.append((d,l,c))

    if len(dtuples) % 10000 == 0:
        print(len(dtuples))

print(f"Bits read = {total_read_bits}; decompresseed tuples = {len(dtuples)}")

assert len(tuples) == len(dtuples)
for i in range(len(tuples)):
    assert dtuples[i][0] == tuples[i][0], f"Decompression failed on tuples {tuples[i]} != {dtuples[i]}"
    assert dtuples[i][1] == tuples[i][1], f"Decompression failed on tuples {tuples[i]} != {dtuples[i]}"
    assert dtuples[i][2] == tuples[i][2], f"Decompression failed on tuples {tuples[i]} != {dtuples[i]}"

res = LZ77_decoder(dtuples)
print(f"LZ77 out : {len(res)} chars; expected {len(GENOME_TEXT)} chars")
assert "".join(res) == GENOME_TEXT


""" Q13. Report the total lengths and compression rates using (a) LZ77
and (b) the combination of LZ77 and Huffman, to encode the genome for
different values of the sliding window size l. Compare your result
with the total length and compression rate obtained using the on-line
Lempel-Ziv algorithm.  Discuss your results. """

for sliding_window_size in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
    # LZ77 only
    tuples = lz77_cached_compression(sliding_window_size, GENOME_TEXT)
    compressed_size, compression_rate = compute_compression_rate_for_LZ77(tuples, sliding_window_size, GENOME_TEXT)

    # LZ77 + Huffman
    bits = lz_with_huffman_encode(sliding_window_size, GENOME_TEXT)
    lz77_huffman_rate = len(GENOME_TEXT)*8 / len(bits)

    print(f"{sliding_window_size} & {compressed_size} & {compression_rate:.2f} & {len(bits)} & {lz77_huffman_rate:.2f}")

exit()




"""
**********************************************************************
Question 11 :
**********************************************************************
11. Famous data compression algorithms combine the LZ77 algorithm and the Huffman algorithm.
Explain how these algorithms can be combined and discuss the interest of the possible combinations.

A/ Using LZ77 first and Huffman second.

https://stackoverflow.com/questions/55547113/why-to-combine-huffman-and-lz77
https://cstheory.stackexchange.com/questions/7653/why-does-huffman-coding-eliminate-entropy-that-lempel-ziv-doesnt
https://www.euccas.me/zlib/

LZ77 is fully deterministic. Sliding window is the only parameter.
LZ77 produces (distance,length,character) tuples.

Huffman : what input shall we give to huffman ?

First porposal : give it the tuples !
=> leads good compression about 5.4 bits per CODON.
BUT the Huffman tree is huge : 100000+ leaves. =>
the TOTAL compressed size = compressed bits + huffman = is actualy bigger than original file.

(d,l,c) -> [ (d,l), c ]

(d,l,c) -> [d] [l] [c]

950KB to about 280KB
c = a CODON; NOT a single character
c = GAT, TAG, CCT, TCA.... NOT c = G, T, C, A

GAT = 3 bytes => 24 bits / per codon
64 codons => log2 64 = 3 bits / per codon

orignal genome size = 952kb (8 bits/cahracter)
with codon letters G,T,C,A (2 bits per charcter) instead of char, we shoudl have 952 / 4 = 240kb
with codon symbols GAT,TCA,ATT,... (6 bits per 3 charcter) instead of char, we shoudl have ((952 / 3)*6)/8 = 238kb
Stc: 265 kb -> results of 3 steps : change alphabet; LZ77; Huffman.
gzip : 280 kb

If one assume there is a set of initial symbols, letters for example, then it is possible
to combine those in bigger symbols if we see repetitions, words for example. Doing, so
we can build a bigger set of symbols $S$ but, in exchange, represent the initial data with
a shorter sequence of symbols $R$. That’s what LZ77 does. It builds a better symbol set,
capturing repetitions. Of course, this makes sense only if we assume that repetitions
are occuring often enough.

So, once the better set of symbols and the associated sequence of symbols are built,
one can use Huffman to compress them. Huffman will build an optimal code book for
representing the symbols given by LZ77.

In practice, the symbols produced by LZ77 are tuples (jump, length, symbol).
Given how many bits we allow to represent each element,  these symbols may
be bigger than what they actually represent. So even if repetitions are caught
by LZ77, we see that the triplets

Given Huffman is parameterless algorithm and leads to an optimal coding, the only variable left is the sliding window
size of LZ77. The bigger the sliding window, the more repetitions (different repeated strings as well as more occurences) LZ77 can find (so bigger $S$, shorter $R$), but,
the deeper the Huffman tree. So a balance must be found.


B/ Using Huffman first and  LZ77 second.





---- quest 11, remaining stuff

1/ run LZ77 directly, then compress each tuple the output with Huffman

"famous" algorithm like

MS-XCA : LZ77, then compress tuple with Huffman
DEFLATE : LZSS (not LZ77!) + canonical Huffman

https://stackoverflow.com/questions/55547113/why-to-combine-huffman-and-lz77

LZ77 est sliding windows mais quid huffman ? => use blocks : cut original file sin blocks



**********************************************************************
Question 12 : what does "the best" mean ? The fastest to compress ? decompress ? The one that produces shortest files ?
**********************************************************************

"""
