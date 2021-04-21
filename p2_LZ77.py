def LZ77_old(input_text, l, look_ahead_length):

    sliding_window = l
    encoded = []
    searchBuffer = ""
    look_ahead_buffer = input_text[:look_ahead_length]
    input_text = input_text[look_ahead_length:]

    while len(look_ahead_buffer) > 0:

        tmpSubstring = look_ahead_buffer[:-1]
        while(len(tmpSubstring) > 1):

            # prefix := longest prefix of input that begins in window
            prefix = findLongestPrefix(searchBuffer, tmpSubstring)
            # if prefix exists in window then
            if(prefix != -1):
                # d := distance to the start of the prefix
                d = len(searchBuffer) - prefix
                # l := length of prefix
                l = len(prefix)

                break

            else:
                # c := first symbol of input
                tmpSubstring = tmpSubstring[0]
                # d and l equal to 0
                d = 0
                l = 0

        # c := char following the prefix in input
        c = look_ahead_buffer[l]

        #append (d, l, c) to encoded input
        encoded.append((d, l, c))

        # shift the sliding window by l + 1 symbols
        sliding_window = l + 1
        # discard l + 1 symbols from the beginning of window and add the l + 1
        # first symbols of the input at the end of the window
        look_ahead_buffer = look_ahead_buffer[sliding_window:]
        look_ahead_buffer = look_ahead_buffer + input_text[:sliding_window]
        input_text = input_text[l + 1:]

    return encoded


def findLongestPrefix(text, char):
    index = 0
    if char in text:
        c = char[0]
        for ch in text:
            if ch == c:
                if text[index:index+len(char)] == char:
                    return index

            index = index + 1

    return -1


def LZ77_decoder(encoded):

    decoded = ""
    for code in encoded:
        d = code[0]
        l = code[1]
        if (d == 0):
            decoded = decoded + code[2]
        else:
            start = len(decoded) - d
            end = start + l
            added = decoded[start:end]
            added = added[:l]
            decoded = decoded + added + code[2]
    return decoded


def LZ77_decoder(encoded):
    decoded = []
    for d, l, c in encoded:
        if l > 0:
            ofs = len(decoded) - d
            for i in range(l):
                decoded.append( decoded[ofs+i])
        decoded.append( c)

    return decoded



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

        r = range(i-1, i-SWSIZE-1, -1)

        for pfx_start in r:

            # Try all prefixes starting at pfx_start
            pfx_len = 0
            prefix = ""

            # Actually, the prefix must fit inside the sliding
            # window and the lookahead must not got past EOF.

            while pfx_start + pfx_len < i and\
                  i+pfx_len < len(input_text) - 1 and\
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

        if longest_prefix_len > 0:
            # print("long")
            d, l, c = i - longest_prefix_pos, longest_prefix_len, input_text[i + longest_prefix_len]
        else:
            d, l, c = 0, 0, input_text[i]

        compressed.append( (d,l,c) )
        #print(f"Ofs:{d}, len:{l}, char:{c}")
        i += l + 1

    return compressed

if __name__ == "__main__":
    S = "abracadabrad"
    print(S)
    print(LZ77_encoder(S, 7))

    import numpy as np
    genome = np.genfromtxt("genome.txt",dtype='str')
    genome = "".join(genome)

    print("Compressing")
    compressed = LZ77_encoder(genome, 1024 // 2)
    print(f"Compression has {len(compressed)} tuples")
    print("Decompressing")
    decompressed = LZ77_decoder(compressed)
    assert decompressed == genome
