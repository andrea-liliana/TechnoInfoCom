import numpy as np


def bin_array(num, m):
    """Convert a positive integer num into a numpy m 0/1 elements vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


class HammingCode:
    def __init__(self, data_bits, code_bits):
        self.data_bits = data_bits
        self.code_bits = code_bits
        self.parity_bits = code_bits - data_bits

        assert code_bits > data_bits
        self._precompute()

    def encode(self, data):
        assert len(data) == self.data_bits
        return (data @ self.G) % 2

    def decode(self, code):
        parity_check = (self.H @ code) % 2
        correction = self.correction_map[np.dot(self.POWER2, parity_check)]
        corrected_code = (code + correction) % 2
        return corrected_code[0:self.data_bits]

    def _precompute(self):
        parities = np.append(
            np.logical_not(np.eye(self.parity_bits, dtype=int)),
            np.ones((1, self.parity_bits), dtype=int),
            axis=0)

        self.G = np.append(np.eye(self.data_bits, dtype=int), parities, axis=1)
        # print("G")
        # print(self.G)

        self.H = np.append(np.transpose(parities),
                           np.eye(self.parity_bits, dtype=int), axis=1)
        print("H")
        print(self.H)

        # Now recovering the data
        self.POWER2 = np.array([2**i for i in range(self.parity_bits)],
                               dtype=int)

        self.correction_map = dict()
        mle_map = dict()  # mle : Most Likely Explanation

        # Compute all possible errors
        for i in range(2**self.code_bits):
            # See:
            #  https://en.wikipedia.org/wiki/Decoding_methods#Syndrome_decoding
            #  https://en.wikipedia.org/wiki/Hamming(7,4)#Error_correction

            # The error that will be added to the correct data
            # (for example during transmission)
            error = bin_array(i, self.code_bits)

            # Compute the effect of the error on the parity check
            h_e = (self.H @ error) % 2
            # print(f"{error} -> {h_e}")

            # Evaluate the most likely explanation given the error
            # To o that we compare the parity check value of that
            # error with identical values given by other errors.
            # The MLE is the error which has the smallest number of
            # wrong bits (bit_in_error).

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
