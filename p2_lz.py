prefixes = {"" : 0}


message = "TGTGTAAACCTTGGTATTGGAATGTAAACACCTTGACACAGAGGTTAGCATTAACACTTAAATGATTAGTTTTTGATCAGTCTATAGATATGGTAGCGTGGAGAGTTTGTGACGGATCCGTGTGGTGAGTGAACAACTACAACTTAGTGTCCGGGAATTCCGGAATCACAGTGTTCGACAGAATACGCGTGGACCGTGGTCAGGAGTATCACAGTGGCGACAAGGACGGGATCTTGAATTGGTAAGAAAATGAACAGTTAGTTGATTCGAATTCATTTGTTTACCTACGTTGTGTACCGG"

coded_message = []

prefix = ""
for c in message:

    if prefix + c in prefixes:
        prefix = prefix + c
    else:
        coded_message.append( [prefixes[prefix], c] )

        prefixes[prefix + c] = len(prefixes)

        prefix = ""


print(len(message))
print(len(coded_message))
print(prefixes)

decoded = ""
prefixes = {0:""}
for prefix_code, c in coded_message:
    prefixes[len(prefixes)] = prefixes[prefix_code] + c
    decoded += prefixes[prefix_code] + c

print(decoded)

assert decoded == message[:-1]
