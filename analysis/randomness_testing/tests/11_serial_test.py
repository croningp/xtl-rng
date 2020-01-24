import numpy as np
import scipy.special as sp

"""
The focus of this test is the frequency of all possible overlapping m-bit patterns across the entire
sequence.  The purpose of this test is to determine whether the number of occurrences of the 2^m m-bit
overlapping patterns is approximately the same as would be expected for a random sequence.  Random
sequences have uniformity; that is, every m-bit pattern has the same chance of appearing as every other
m-bit pattern.

Note that if delta^2psi^2_m or deltapsi^2_m had been large then non-uniformity of the m-bit blocks is implied.
"""

MIN_BITS = 10

class Serial:

    def __init__(self, bits, p_threshold = 0.01):

        self.bits = bits
        self.p_threshold = p_threshold
        self.n = len(bits)

        self.psi0 = None
        self.psi1 = None
        self.psi2 = None

        self.d1 = None
        self.d2 = None

        self.p1 = None
        self.p2 = None

        self.p = None           # p value
        self.success = None
        self.test_run = False

        self.check_enough_bits()

        if self.enough_bits: self.run_test()

    def check_enough_bits(self):
        self.enough_bits = MIN_BITS <= self.n

    def run_test(self):
        # self.m = max(int(np.floor(np.log2(self.n)))-8, 6)
        self.m=16
        padded_bits = self.bits + self.bits[0:self.m-1]

        self.psi0 = self.psi_sq_mv1(self.m, padded_bits)
        self.psi1 = self.psi_sq_mv1(self.m-1, padded_bits)
        self.psi2 = self.psi_sq_mv1(self.m-2, padded_bits)
        # self.psi2 = 0

        self.d1 = self.psi0 - self.psi1
        self.d2 = self.psi0 - 2*self.psi1 + self.psi2

        self.p1 = sp.gammaincc(2**(self.m-2), self.d1/2)
        self.p2 = sp.gammaincc(2**(self.m-3), self.d2/2)
        self.p = self.p1
        self.success = (self.p >= 0.01)
        self.test_run = True


    def psi_sq_mv1(self, m, padded_bits):
        counts = [0 for i in range(2**m)]
        for idx in range(self.n):
            block = padded_bits[idx:idx+m]
            block_as_int = int(''.join([str(i) for i in block]), 2)
            counts[block_as_int] += 1

        psi_sq_m = sum([c**2 for c in counts])*(2**m)/self.n-self.n

        return psi_sq_m

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import utils

    e = '../numbers/data.e'
    bits = utils.from_file(e)[:1000000]
    # bits = utils.from_string('0011011101')
    S = Serial(bits)
    print(S.p2) # p-value = 0.56195
