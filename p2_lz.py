import os
import math
import heapq
from io import StringIO
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from p2_LZ77 import LZ77_encoder, LZ77_decoder

INPUT_FILE = "genome.txt"
CODON_LEN = 3


""" Q1 Implement a function that returns a binary Huffman code for a given probability distribution. Give
the main steps of your implementation. Explain how to extend your function to generate a Huffman
code of any alphabet size. Verify your code on Exercise 7 of the second list of exercises, and report
the output of your code for this example. """


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


""" Q2. Given a sequence of symbols, implement a function that returns
a dictionary and the encoded sequence using the on-line Lempel-Ziv
algorithm (see State of the art in data compression, slide
50/53). Reproduce and report the example given in the course.

"""

def bits_to_represent(nb_values):
    # Disontinuous ! f == 0 in x==1 !
    assert nb_values >= 1
    return math.ceil(math.log2(nb_values))

def code_binary_char(c):
    return c

def code_ascii_char(c):
    return f"{ord(c):08b}"

def online_lz_compress(data, code_char):
    coded_bin = ""

    prefix = ""
    prefixes = {"": 0}

    EOF_MARK = ''

    # The whole point of this trickery is to detect
    # EOF one char ahead. This way we can tell when
    # we're processing the last character of the
    # input file

    c = data.read(1)  # Read one byte
    assert c != EOF_MARK, "Compressing empty file is not supported"
    next_char = data.read(1)
    eof = next_char == EOF_MARK

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
            coded_bin += f"{pfx}{char}"

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

    return coded_bin

def decode_binary_char(data, ndx):
    return data[ndx], 1

def decode_ascii_char(data, ndx):
    return chr(int(data[ndx:ndx+8], 2)), 8

def online_lz_decompress(coded_bin, decode_char):
    ndx = 0
    decoded = ""
    prefixes = {0: ""}
    while ndx < len(coded_bin):

        l = bits_to_represent(len(prefixes))
        if l > 0:
            prefix_code = int(coded_bin[ndx:ndx+l], 2)
        else:
            prefix_code = 0

        char, skips = decode_char(coded_bin, ndx+l)

        decoded += prefixes[prefix_code] + char
        prefixes[len(prefixes)] = prefixes[prefix_code] + char
        #ndx = ndx+l+8
        ndx = ndx+l+skips

    return decoded


from bitarray import bitarray

def int32_to_bits(i):
    return bitarray(f"{i:032b}")

def compress_values(values):
    bits_per_value = bits_to_represent(max(values)+1) # +1 is tricky ! We assume we encode values 0 to max ! FIXME !
    print(f"Compress: bits_per_value={bits_per_value}, nb_values={len(values)} [{min(values)}, {max(values)}]")

    bits = bitarray()
    bits.extend(int32_to_bits(bits_per_value))
    bits.extend(int32_to_bits(len(values)))

    fmt = f"{{:0{bits_per_value}b}}"
    for v in values:
        bits.extend(bitarray(fmt.format(v)))
    return bits

def decompress_values(bits):
    bits_per_value = int(bits[0:32].to01(), 2)
    nb_values = int(bits[32:64].to01(), 2)

    print(f"Decompress: bits_per_value={bits_per_value}, nb_values={nb_values}")
    # Return bits read, decompressed counts dictionary
    return 64 + bits_per_value*nb_values, \
        [int(bits[i:i+bits_per_value].to01(), 2)
         for i in range(64,64+nb_values*bits_per_value, bits_per_value) ]


def compress_counts(counts):
    # for v in counts.keys():
    #     assert v > 0, "0 is reserved for end of stream symbol"

    bits = bitarray()
    bits.extend(compress_values(counts.keys()))
    bits.extend(compress_values(counts.values()))
    return bits

def decompress_counts(bits):
    total_read_bits = 0
    read_bits, k = decompress_values(bits[total_read_bits:])
    total_read_bits += read_bits
    read_bits, v = decompress_values(bits[total_read_bits:])
    total_read_bits += read_bits

    # Return bits read, decompressed counts dictionary
    return total_read_bits, dict(zip(k,v))

# Test compress values
bits = compress_values([1,2,3,4,5,6,7,8,9,10])
decomp_bits, values = decompress_values(bits)
assert decomp_bits == len(bits)

# Test compress counts
bits = compress_counts({1:2, 10:11, 12:256})
decomp_bits, cmap = decompress_counts(bits)
assert decomp_bits == len(bits), f"{decomp_bits} != {len(bits)}"


if False:
    slide_50 = "1 0 11 01 010 00 10".replace(" ", "")
    compressed = online_lz_compress(StringIO(slide_50), code_binary_char)
    assert compressed == "1 00 011 101 1000 0100 0010".replace(" ", "")
    assert online_lz_decompress(compressed, decode_binary_char) == slide_50

    with open(INPUT_FILE, 'r') as genome:
        compressed = online_lz_compress(genome, code_ascii_char)

    with open(INPUT_FILE, 'r') as genome:
        assert online_lz_decompress(compressed, decode_ascii_char) == genome.read()


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


# class StringStore:
#     def __init__(self):
#         self._store = StringIO()

#     def write(self, symbol)):
#         self._store.write(symbol)

#     def value(self):
#         return self._store.getvalue()


# class SymbolStore:
#     def __init__(self):
#         self._store = StringIO()

#     def write(self, symbol)):
#         self._store.write(symbol)

#     def value(self):
#         return self._store.getvalue()

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

def decode(genome, decode_map, nb_symbols = 2**31):
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


""" Q5. Estimate the marginal probability distribution of all codons
from the given genome, and determine the corresponding binary Huffman
code and the encoded genome.  Give the total length of the encoded
genome and the compression rate. """

genome = "".join(np.genfromtxt("genome.txt", dtype='str'))
codons_cnt = Counter(codons_iterator(genome))
CODONS = sorted(codons_cnt.keys())

if True:

    top_node = build_huffman_tree(codons_cnt)
    code_map, decode_map = build_codebooks(top_node)
    compressed = encode(codons_iterator(genome), code_map)
    # Validate that compression works by decompressing
    assert genome == decode(
        genome, decode_map), "Decompressed data is not the same as compressed data"

    ratio = len(compressed) / (len(genome)*8)
    print(f"Genome size = {len(genome)*8} bits Compressed size = {len(compressed)} bits; ratio={ratio}")


""" For Q7 of Project2, I have two questions.

1: Must we reuse the same Huffman codes (those of Q5) for each input
genome length or should we recompute the Huffman codes for each of the
input genome length ? I ask because I don't see the point in reusing
the same Huffman code all the time.

2: You ask to plot the empirical average length of the encoded genome.
Do you confirm that there's an average to compute ? Indeed, when I
encode the genome, there's only one possible encoding with one length,
so I don't see on what I must compute an average.
"""

if False:
    STEP = len(genome)//10
    for i in range(STEP, len(genome), STEP):
        print(i)
        g = genome[0:i]
        codons_cnt = Counter(codons_iterator(genome))
        top_node = build_huffman_tree(codons_cnt)
        code_map, decode_map = build_codebooks(top_node)

        compressed = encode(g, code_map)
        assert g == decode(compressed, decode_map)
        print(f"{len(compressed)} / {len(g)*8} = {len(compressed) / (len(g)*8):.3f}")


""" Q12. Encode the genome using the best (according to your answer in
the previous question) combination of LZ77 and Huffman
algorithms. Give the total length of the encoded genome and the
compression rate.
"""

def tuples_iterator(tuples):
    for t in tuples:
        yield t

#genome=genome[:16*CODON_LEN]

# Split the genom into codons.
# FIXME Try splitting the genome into letters : G,T,C,A
# But not correct yet to compare to gzip : G,T,C,A will be 4 symbols => 2 bits per letter :-)
# But gzip doesn't know that, it sees bytes => 8 bits per letter !

# G  A  T  A  C  A GT ACAGT GTGAACCTA
# 00 01 11 00 10 01
# G,A,T,A,A,G,TACA
#

genome_as_codons = [c for c in codons_iterator(genome)]
print(f"{len(genome_as_codons)} codons")

import os.path
import pickle

""" Q10. Encode the genome using the LZ77 algorithm. Give the total
length of the encoded genome and the compression rate."""

WIN_SIZE = 512*2
CACHE_NAME=f"LZ77Cache{WIN_SIZE}.dat"
if not os.path.exists(CACHE_NAME):
    print("Crunching with LZ77")
    tuples = LZ77_encoder(genome_as_codons, WIN_SIZE)
    with open(CACHE_NAME,"wb") as fout:
        pickle.dump(tuples, fout)
else:
    with open(CACHE_NAME,"rb") as fin:
        tuples = pickle.load(fin)

print("Decrunching with LZ77")
res = LZ77_decoder(tuples)
print(f"Length decompressed={len(res)}, expected={len(genome)}")
assert "".join(res) == genome, "LZ77 compression wen wrong"


"""Q12. Encode the genome using the best (according to your answer in
the previous question) combination of LZ77 and Huffman
algorithms. Give the total length of the encoded genome and the
compression rate."""


compressed_size = len(tuples)*(10+10+6) # 10 bits for distance, 10 bits for length, 6 bits for codon.
print(f"LZ77 only : {len(tuples)} tuples -> {compressed_size} bits => {compressed_size / len(genome_as_codons):.1f} bits per codon")

small_c = [c for d,l,c in tuples]
small_l = [l for d,l,c in tuples]
small_d = [d for d,l,c in tuples]

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
print("Compressed size = {} bits => {:.1f} bits per codons".format(compressed_size, compressed_size / len(genome_as_codons)))

# --------------------------------------------------------------------
# (l,d,c) -> (l,c) + (d)

dist_count = Counter(small_d) # 1 -> 001 ,2->110,3->1110,2,5,1,1,1,1,1,4,6,...,1000 -> dict( symbol -> count)
len_count = Counter(small_l)

c = Counter(small_c)
char_count = dict(zip(CODONS, [c[codon] for codon in CODONS]))

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
char_code_map, decode_map = build_codebooks(top_node)

print(f"Dist counts : {len(dist_count)}")
print(f"Len counts : {len(len_count)}")
print(f"C counts : {len(char_count)}")

print(f"1. l + c + d : build_huffman_tree on {len(tuples)} tuples (of which {len(char_count)} are unique)")

compressed_size = sum([len(char_code_map[c]) + len(len_code_map[l]) + len(dist_code_map[d]) for d,l,c in tuples]) + all_trees_size
print("Compressed size = {} bits {} bytes => {:.1f} bits per codons".format(compressed_size, compressed_size//8, compressed_size / len(genome_as_codons)))
plt.figure()
plt.plot(list(sorted(char_count.values())))
plt.show()


# --------------------------------------------------------------------

def tuples_value_iterator(tuples, idx):
    for t in tuples:
        yield t[idx]

# Compression ************************************************

bits = bitarray()
bits.extend(compress_counts(dist_count))
bits.extend(compress_counts(len_count))

# This expression ensure the {codon -> count} map is
# sorted in codons order. This willl esae the decompression
# And prevent us to store the codons values themselves.

assert len(char_count) == 64, f"FIXME Simplifcation, all codons must show up, I count {len(char_count)} out of {4**CODON_LEN}"
bits.extend(
    compress_values(
        [char_count[codon] for codon in CODONS]))

LETTER = ['A','T','C','A']

# for i, code_map in enumerate([dist_code_map, len_code_map, char_code_map]):
#     binstring = encode(tuples_value_iterator(tuples, i), code_map)
#     bits.extend(bitarray(binstring))


for d,l,c in tuples:
    # Compress a tuple (d,l,c) into its huffman representation
    # (d -> 0101..., l -> 11000, c->101010), using a different
    # codebook for d,l and c.
    bits.extend(bitarray(dist_code_map[d]))
    bits.extend(bitarray(len_code_map[l]))
    bits.extend(bitarray(char_code_map[c]))

print(f"{len(bits)} bits => {len(bits)//8} bytes")


# Decompression ************************************************

total_read_bits = 0
read_bits, dist_count = decompress_counts(bits)
total_read_bits += read_bits
read_bits, len_count = decompress_counts(bits[total_read_bits:])
total_read_bits += read_bits
read_bits, char_count = decompress_values(bits[total_read_bits:])
char_count = dict(zip(CODONS, char_count))
total_read_bits += read_bits

print(len(dist_count))
print(len(len_count))
print(len(char_count))

assert sum(dist_count.values()) == sum(len_count.values()) == sum(char_count.values()) == len(tuples)

top_node = build_huffman_tree(dist_count)
dist_code_map, dist_decode_map = build_codebooks(top_node)
top_node = build_huffman_tree(len_count)
len_code_map, len_decode_map = build_codebooks(top_node)
top_node = build_huffman_tree(char_count)
char_code_map, char_decode_map = build_codebooks(top_node)

dtuples = []
for i in range(sum(dist_count.values())):
    # total_read_bits+100 : make sure we don't pass all the remaining
    # of bit array to decode_one_symbol function each time => it's
    # speed optimisation.

    read_bits, d = decode_one_symbol(bits[total_read_bits:total_read_bits+100], dist_decode_map)
    total_read_bits += read_bits
    read_bits, l = decode_one_symbol(bits[total_read_bits:total_read_bits+100], len_decode_map)
    total_read_bits += read_bits
    read_bits, c = decode_one_symbol(bits[total_read_bits:total_read_bits+100], char_decode_map)
    total_read_bits += read_bits
    dtuples.append((d,l,c))

    if len(dtuples) % 10000 == 0:
        print(len(dtuples))
print(total_read_bits)
print(len(dtuples))

res = LZ77_decoder(dtuples)
print(len(res))
print(len(genome))

assert len(tuples) == len(dtuples)
for i in range(len(tuples)):
    #print(dtuples[i], tuples[i])
    assert dtuples[i][0] == tuples[i][0]
    assert dtuples[i][1] == tuples[i][1]
    assert dtuples[i][2] == tuples[i][2], f"{tuples[i]} != {dtuples[i]}"

assert "".join(res) == genome


exit()


""" Q9. Encode the genome using the on-line Lempel-Ziv algorithm. Give
the total length of the encoded genome and the compression rate."""

# Online LZ algorithm

coded_message = []
coded_bin = ""

prefix = ""
prefixes = {"": 0}

EOF_MARK = ''

with open(INPUT_FILE, 'r') as genome:

    # The whole point of this trickery is to detect
    # EOF one char ahead. This way we can tell when
    # we're processing the last character of the
    # input file

    c = genome.read(1)  # Read one byte
    assert c != EOF_MARK, "Compressing empty file is not supported"
    next_char = genome.read(1)
    eof = next_char == EOF_MARK

    while True:
        if not eof and prefix + c in prefixes:
            prefix = prefix + c
        else:
            # Either EOF on next char,
            # either prefix+c not in prefixes.

            # In both case we store the prefix+c. That's
            # especially useful when EOF is met since it
            # allows us to termiante the output stream cleanly.

            # Number of bits necessary to represent the length
            # of the prefixes table.
            # FIXME this is not really fast.
            l = math.ceil(math.log2(len(prefixes)))
            if l == 0:
                l = 1

            # Append (prefix's number, c) to the output stream .
            coded_bin += f"{np.binary_repr(prefixes[prefix],l)}{ord(c):08b}"

            # Record the new prefix and give it a number
            prefixes[prefix + c] = len(prefixes)
            prefix = ""

        if next_char != EOF_MARK:
            c = next_char
            next_char = genome.read(1)
            eof = next_char == EOF_MARK
        else:
            break

print(os.path.getsize(INPUT_FILE))

# Test : from 0/1 string to np array
as_bin = np.frombuffer(np.array(map(int, coded_bin)), np.uint8)
print(as_bin)
print(np.unpackbits(as_bin)[0:len(coded_bin)])

print(len(coded_bin) // 8)
# print(coded_bin)

ndx = 0
decoded = ""
prefixes = {0: ""}
while ndx < len(coded_bin):
    l = math.ceil(math.log2(len(prefixes)))
    if l == 0:
        l = 1
    prefix_code = int(coded_bin[ndx:ndx+l], 2)

    #print(f"ndx:{ndx}, l:{l}, len pfx:{len(prefixes)}, pfx code:{prefix_code}")
    c = chr(int(coded_bin[ndx+l:ndx+l+8], 2))
    # c = coded_bin[ndx+l:ndx+l+1]

    decoded += prefixes[prefix_code] + c
    # print(l,prefix_code,c,decoded)

    prefixes[len(prefixes)] = prefixes[prefix_code] + c
    ndx = ndx+l+8

# decoded = ""
# prefixes = {0: ""}
# for prefix_code, c in coded_message:
#     prefixes[len(prefixes)] = prefixes[prefix_code] + c
#     decoded += prefixes[prefix_code] + c

# print(decoded)

with open(INPUT_FILE, 'r') as genome:
    assert decoded == genome.read()


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
