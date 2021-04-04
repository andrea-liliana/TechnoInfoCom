import os
import numpy as np
import math

INPUT_FILE = "genome.txt"


coded_message = []
coded_bin = ""

prefix = ""
prefixes = {"" : 0}


with open(INPUT_FILE, 'r') as f:

    # The whole point of this reickery is to detect
    # EOF one char ahead. This way we can tell when
    # we're processing the last character of the
    # input file

    c = f.read(1)
    eof = c == ''
    next_char = f.read(1)

    while True:
        if not eof and prefix + c in prefixes:
            prefix = prefix + c
        else:
            # coded_message.append( [prefixes[prefix], c] )

            # FIXME one must optimize this
            l = math.ceil(math.log2(len(prefixes)))
            if l == 0:
                l = 1

            coded_bin += f"{np.binary_repr(prefixes[prefix],l)}{ord(c):08b}"

            prefixes[prefix + c] = len(prefixes)
            prefix = ""

        if next_char != '':
            c = next_char
            next_char = f.read(1)
            eof = next_char == ''
        else:
            break

print(os.path.getsize(INPUT_FILE))

# Test : from 0/1 string to np array
print(np.frombuffer(np.array(map(int, coded_bin)), np.uint8))

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

with open(INPUT_FILE, 'r') as f:
    assert decoded == f.read()
