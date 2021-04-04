import math

message = "TGTGTAAACCTTGGTATTGGAATGTAAACACCTTGACACAGAGGTTAGCATTAACACTTAAATGATTAGTTTTTGATCAGTCTATAGATATGGTAGCGTGGAGAGTTTGTGACGGATCCGTGTGGTGAGTGAACAACTACAACTTAGTGTCCGGGAATTCCGGAATCACAGTGTTCGACAGAATACGCGTGGACCGTGGTCAGGAGTATCACAGTGGCGACAAGGACGGGATCTTGAATTGGTAAGAAAATGAACAGTTAGTTGATTCGAATTCATTTGTTTACCTACGTTGTGTACCGG"

coded_message = []
coded_bin = ""

prefix = ""
prefixes = {"" : 0}
for c in message:

    if prefix + c in prefixes:
        prefix = prefix + c
    else:
        coded_message.append( [prefixes[prefix], c] )

        coded_bin += f"X{prefixes[prefix]:b}{ord(c):8b}"

        prefixes[prefix + c] = len(prefixes)

        prefix = ""


print(len(message))
print(len(coded_message))
print(coded_bin)

ndx = 0
decoded = ""
prefixes = {0: ""}
while True:
    l = math.ceil(math.log2(len(prefixes)))
    if l == 0:
        l = 1
    prefix_code = int(coded_bin[ndx:ndx+l], 2)
    print(f"ndx:{ndx}, l:{l}, len pfx:{len(prefixes)}, pfx code:{prefix_code}")
    c = chr(int(coded_bin[ndx+l:ndx+l+8], 2))
    decoded += prefixes[prefix_code] + c
    print(l,prefix_code,c)

    prefixes[len(prefixes)] = prefixes[prefix_code] + c
    ndx = ndx+l+8



# decoded = ""
# prefixes = {0: ""}
# for prefix_code, c in coded_message:
#     prefixes[len(prefixes)] = prefixes[prefix_code] + c
#     decoded += prefixes[prefix_code] + c

print(decoded)

assert decoded == message[:-1]
