import os
import math
import heapq
from io import StringIO
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from p2_LZ77 import LZ77_encoder

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


def build_huffman_tree(codons_cnt: dict):
    # Create leaves of the tree
    nodes = []
    for codon, cnt in codons_cnt.items():
        nodes.append((cnt, Node(None, None, cnt, codon)))

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


ex7_freq = [0.05, 0.10, 0.15, 0.15, 0.2, 0.35]
symbols = [f"{x:.2f}" for x in ex7_freq]
top_node = build_huffman_tree(dict(zip(symbols, ex7_freq)))
leaves = compute_leaves_codes(top_node)


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
    # Disontinuous ! == 0 in x==1 !
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

def compress_values(values):
    m = bits_to_represent(max(values))
    fmt = f"{{:0{m}b}}"
    bits = bitarray()
    for v in values:
        bits.extend(bitarray(fmt.format(v)))
    return bits, m

def decompress_values(bits, nb_values, bits_per_value):
    return [int(bits[i*bits_per_value:(i+1)*bits_per_value].to01(), 2) for i in range(nb_values) ]

def int32_to_bits(i):
    return bitarray(f"{i:032b}")

def compress_counts( counts):
    bits_k,btr_k = compress_values(counts.keys())
    bits_v,btr_v = compress_values(counts.values())

    bits = int32_to_bits(btr_k)
    bits.extend(int32_to_bits(btr_v))
    bits.extend(int32_to_bits(len(counts)))
    bits.extend(bits_k)
    bits.extend(bits_v)
    return bits

def decompress_counts(bits):
    btr_k = int(bits[0:32].to01(),2)
    btr_v = int(bits[32:64].to01(),2)
    length = int(bits[64:96].to01(),2)

    nk = btr_k * length
    nv = btr_v * length

    k = [int(bits[96+i:96+i+btr_k].to01(),2) for i in range(0,nk, btr_k)]
    v = [int(bits[96+i:96+i+btr_k].to01(),2) for i in range(nk,nk+nv, btr_v)]

    return dict(zip(k,v))

print(compress_values([1,2,3]))
print(decompress_values(compress_values([1,2,3])[0], 3, 2))
print(decompress_counts(compress_counts({1:2, 10:11, 12:13})))


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
    """ Convert a serie of symbol into a serie of
    corresponding codewords.

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


def decode(genome, decode_map):
    # File end is detected by file size. See remark in
    # encode() funtcion.

    prefix = ""
    file_str = StringIO()
    for c in compressed:
        prefix += c
        if prefix in decode_map:
            file_str.write(decode_map[prefix])
            prefix = ""
    return file_str.getvalue()


""" 5. Estimate the marginal probability distribution of all codons from the given genome, and determine
the corresponding binary Huffman code and the encoded genome.
Give the total length of the encoded genome and the compression rate. """

genome = "".join(np.genfromtxt("genome.txt", dtype='str'))
if True:
    codons_cnt = Counter(codons_iterator(genome))

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

#genome=genome[:1000*CODON_LEN]

genome_as_codons = [c for c in codons_iterator(genome)]
print(f"{len(genome_as_codons)} codons")

import os.path
import pickle

WIN_SIZE = 512*4
CACHE_NAME=f"LZ77Cache{WIN_SIZE}.dat"
if not os.path.exists(CACHE_NAME):
    tuples = LZ77_encoder(genome_as_codons, WIN_SIZE)
    with open(CACHE_NAME,"wb") as fout:
        pickle.dump(tuples, fout)
else:
    with open(CACHE_NAME,"rb") as fin:
        tuples = pickle.load(fin)

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


tree_size = len(tuples_count) * (bits_to_represent(len(small_c)) + bits_to_represent(max(small_d)) + bits_to_represent(max(small_l)) + bits_to_represent(len(tuples_count)))
print(f"Tree size = {tree_size} bits => {tree_size//8} bytes")


top_node = build_huffman_tree(tuples_count)
code_map, decode_map = build_codebooks(top_node)
compressed_size = sum([len(code_map[t])*cnt for t, cnt in tuples_count.items()]) + tree_size
print("Compressed size = {} bits => {:.1f} bits per codons".format(compressed_size, compressed_size / len(genome_as_codons)))

# --------------------------------------------------------------------
# (l,d,c) -> (l,c) + (d)

dist_count = Counter(small_d)
len_count = Counter(small_l)
char_count = Counter(small_c)

plt.figure()
plt.plot(list(dist_count.values()))
plt.xlabel("Distances")
plt.ylabel("# occurences")
plt.show()

plt.figure()
plt.plot(list(len_count.values()))
plt.xlabel("Lengths")
plt.ylabel("# occurences")
plt.show()

top_node = build_huffman_tree(dist_count)
dist_code_map, dist_decode_map = build_codebooks(top_node)

dist_tree_size = len(dist_count) * (bits_to_represent(max(dist_count.values())) + bits_to_represent(max(dist_count.keys())))
len_tree_size = len(len_count) * (bits_to_represent(max(len_count.values())) + bits_to_represent(max(len_count.keys())))
char_tree_size = len(char_count) * bits_to_represent(max(char_count.values()))

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

bits = bitarray()
bits.extend(compress_counts(dist_count))
bits.extend(compress_counts(len_count))

# This expression ensure the {codon -> count} map is
# sorted in codons order. This willl esae the decompression
# And prevent us to store the codons values themselves.

assert len(char_count) == 64, f"FIXME Simplifcation, all codons must show up {len(char_count)}"
bits.extend(
    compress_values(
        [cnt for codon, cnt in sorted(
            [(codon, cnt) for codon, cnt in char_count.items()])]))

bits.extend(encode(codons_iterator(genome), char_code_map))

print(f"{len(bits)} bits => {len(bits)//8} bytes")

exit()


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

Hello, I have a question regarding question 11. The famous algorithms your talking about are not many. The closest I could find are DEFLATE and zlib (both pretty dated, see LZSA, LZ4, etc). Basically many algorithms use a combination of *variants* of LZ77 (LZ78, LZW,...) and *variants* of Huffman encoding (simple encoding, canonical encoding, double encoding (see deflate), etc.). So I wanted to make it 100% clear that you require us to study how to combine the LZ77 variant (as proposed in the project statement) and the regular Huffman tree construction (as explained in the course), and only those two. Is it correct ?



A/ Using LZ77 first and Huffman second.

https://stackoverflow.com/questions/55547113/why-to-combine-huffman-and-lz77
https://cstheory.stackexchange.com/questions/7653/why-does-huffman-coding-eliminate-entropy-that-lempel-ziv-doesnt
https://www.euccas.me/zlib/

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
