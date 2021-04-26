from io import StringIO
import numpy as np
from p2_utils import bits_to_represent


def code_binary_char(c):
    return c


def code_ascii_char(c):
    return f"{ord(c):08b}"


def online_lz_compress(data, code_char, tuples_out=False):

    # The whole point this code below is to detect
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
        ndx = ndx+l+skips

    return decoded.getvalue()
