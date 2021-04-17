import os
import math
import heapq
from io import StringIO
import numpy as np

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
        nodes.append((cnt,Node(None, None, cnt, codon)))

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

with open("graph.dot","w") as fout:
    fout.write("graph HuffmanTree {\n")
    draw_neato_tree(fout,top_node)
    r = " ".join([ gid(n)+";" for n in leaves])
    fout.write(f"{{rank = same; {r}}}\n")
    fout.write("}\n")
exit()


def build_codebooks(top_node):
    # Affect a code to each leaf node
    d = compute_leaves_codes(top_node, "")

    # Build maps from/to symbol to/from Huffman codes
    code_map = dict()
    decode_map = dict()
    for node in sorted(d, key=lambda n:n.weight):
        #print(f"{node.symbol} {node.weight:5d} {node.code}")
        code_map[node.symbol] = node.code
        decode_map[node.code] = node.symbol

    return code_map, decode_map


def compute_codons_frequencies(genome):
    codons_cnt = dict()
    for i in range(0, len(genome), CODON_LEN):
        codon = genome[i:i+CODON_LEN]
        if codon not in codons_cnt:
            codons_cnt[codon] = 1
        else:
            codons_cnt[codon] += 1
    return codons_cnt

def encode(genome, code_map):
    file_str = StringIO()
    for i in range(0, len(genome), CODON_LEN):
        codon = genome[i:i+CODON_LEN]
        file_str.write(code_map[codon])

    return file_str.getvalue()


def decode(genome, decode_map):
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

genome = np.genfromtxt("genome.txt", dtype='str')
genome = "".join(genome)
codons_cnt = compute_codons_frequencies(genome)
top_node = build_huffman_tree(codons_cnt)
code_map, decode_map = build_codebooks(top_node)
compressed = encode(genome, code_map)
# Validate that compression works by decompressing
assert genome == decode(genome, decode_map), "Decompressed data is not the same as compressed data"

print(f"Genome size = {len(genome)*8} bits Compressed size = {len(compressed)} bits")


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

STEP = len(genome)//10
for i in range(STEP, len(genome), STEP):
    print(i)
    g = genome[0:i]
    codons_cnt = compute_codons_frequencies(g)
    top_node = build_huffman_tree(codons_cnt)
    code_map, decode_map = build_codebooks(top_node)

    compressed = encode(g, code_map)
    assert g == decode(compressed, decode_map)
    print(f"{len(compressed)} / {len(g)*8} = {len(compressed) / (len(g)*8):.3f}")

exit()


# Online LZ algorithm

coded_message = []
coded_bin = ""

prefix = ""
prefixes = {"" : 0}

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
    #print(l,prefix_code,c,decoded)

    prefixes[len(prefixes)] = prefixes[prefix_code] + c
    ndx = ndx+l+8

# decoded = ""
# prefixes = {0: ""}
# for prefix_code, c in coded_message:
#     prefixes[len(prefixes)] = prefixes[prefix_code] + c
#     decoded += prefixes[prefix_code] + c

#print(decoded)

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
