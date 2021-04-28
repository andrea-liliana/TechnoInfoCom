import numpy as np


def bin_array(num, m):
    """Convert a positive integer num into a numpy m 0/1 elements vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


class HammingCode:
    def __init__(self, data_bits, code_bits):
        """ For example for hamming(4,7) : data_bits=4, code_bits=7.
        """
        assert code_bits > data_bits, "Not a valid Hamming configuration"

        self._data_bits = data_bits
        self._code_bits = code_bits
        self._parity_bits = code_bits - data_bits

        self._precompute()

    def encode(self, word):
        """ Encode word acording to the Hamming code.
        The word is a numpy array of 1 and 0.
        Return the encoded word as a numpy array of 1 and 0.
        """
        return (word @ self.G) % 2

    def decode(self, codeword):
        # Decode codeword (a numpy array of 0 and ones)
        # Compute the vector (data, parity) bits of the received codeword.
        parity_check = (self.H @ codeword) % 2

        # Convert the vector of data+parity bits into a number
        # and use it to find the correction to apply (if any)
        # (equivalent to the most likely error given codeword)
        correction = self.correction_map[np.dot(self.POWER2, parity_check)]

        # Apply the correction (equivalently : undo the error)
        corrected_code = (codeword + correction) % 2
        return corrected_code[0:self._data_bits]

    def _precompute(self):
        # Compute useful matrices according to
        # https://en.wikipedia.org/wiki/Hamming(7,4)
        parities = np.append(
            np.logical_not(np.eye(self._parity_bits, dtype=int)),
            np.ones((1, self._parity_bits), dtype=int),
            axis=0)

        self.G = np.append(np.eye(self._data_bits, dtype=int), parities, axis=1)

        self.H = np.append(np.transpose(parities),
                           np.eye(self._parity_bits, dtype=int), axis=1)

        # Now work out the way to correct the received codewords.
        self.POWER2 = np.array([2**i for i in range(self._parity_bits)],
                               dtype=int)

        self.correction_map = dict()
        mle_map = dict()  # mle : Most Likely Explanation

        # Compute all possible errors
        for i in range(2**self._code_bits):
            # See:
            #  https://en.wikipedia.org/wiki/Decoding_methods#Syndrome_decoding
            #  https://en.wikipedia.org/wiki/Hamming(7,4)#Error_correction

            # Simulate the error that could be added to the correct data
            # (for example during transmission). A one means error,
            # a zero means none.
            error = bin_array(i, self._code_bits)

            # Compute the effect of the error on the parity check
            h_e = (self.H @ error) % 2

            # Evaluate the most likely explanation given the error
            # To do that we compare the parity check value of that
            # error with identical values given by other errors.
            # The MLE is the error which has the smallest number of
            # wrong (1) bits (bit_in_error).

            ndx = np.dot(self.POWER2, h_e)
            bits_in_error = np.sum(error)
            if ndx not in mle_map or bits_in_error < mle_map[ndx]:
                mle_map[ndx] = bits_in_error

                # Remember the most likely error associated to a given
                # value of the parity check.
                self.correction_map[ndx] = error


if __name__ == "__main__":
    hamming = HammingCode(4, 7)

    # Check that all hamming codes decode properly
    # (without error)
    for i in range(2**4):
        data = bin_array(i, 4)
        check = hamming.decode(hamming.encode(data))
        assert np.array_equal(check, data),  f"{data} != {check}"

    # Introduce one bit of error in the data bits
    for i in range(2**4):
        for err in range(4):
            data = bin_array(i, 4)
            code = hamming.encode(data)
            code[err] = 1 ^ code[err]
            check = hamming.decode(code)
            assert np.array_equal(check, data),\
                f"{data} != {check}, error ndx={err}"

    # Introduce one bit of error in the parity bits
    for i in range(2**4):
        for err in range(4, 7):
            data = bin_array(i, 4)
            code = hamming.encode(data)
            code[err] = 1 ^ code[err]
            check = hamming.decode(code)
            assert np.array_equal(check, data),\
                f"{data} != {check}, error ndx={err}"
