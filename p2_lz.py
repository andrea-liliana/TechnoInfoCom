import numpy as np
import math

message = "TGTGTAAACCTTGGTATTGGAATGTAAACACCTTGACACAGAGGTTAGCATTAACACTTAAATGATTAGTTTTTGATCAGTCTATAGATATGGTAGCGTGGAGAGTTTGTGACGGATCCGTGTGGTGAGTGAACAACTACAACTTAGTGTCCGGGAATTCCGGAATCACAGTGTTCGACAGAATACGCGTGGACCGTGGTCAGGAGTATCACAGTGGCGACAAGGACGGGATCTTGAATTGGTAAGAAAATGAACAGTTAGTTGATTCGAATTCATTTGTTTACCTACGTTGTGTACCGG"

coded_message = []
coded_bin = ""

prefix = ""
prefixes = {"" : 0}
for i, c in enumerate(message):

    if i < len(message) - 1 and prefix + c in prefixes:
        prefix = prefix + c
    else:
        coded_message.append( [prefixes[prefix], c] )

        l = math.ceil(math.log2(len(prefixes)))
        if l == 0:
            l = 1

        coded_bin += f"{np.binary_repr(prefixes[prefix],l)}{ord(c):08b}"

        prefixes[prefix + c] = len(prefixes)

        prefix = ""

# if prefix != "":
#     l = math.ceil(math.log2(len(prefixes)))
#     if l == 0:
#         l = 1

#     coded_bin += f"{np.binary_repr(prefixes[prefix],l)}{ord(c):08b}"


print(len(message))
print(len(coded_message))
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

print(decoded)

assert decoded == message
