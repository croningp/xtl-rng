import numpy as np
import scipy.special as sp

"""
As with the Serial test of Section 2.11, the focus of this test is the frequency of all possible  overlapping
m-bit patterns across the entire sequence.  The purpose of the test is to compare the frequency of
overlapping blocks of two consecutive/adjacent lengths (m and m+1) against the expected result for a
random sequence.

Note that small values of ApEn(m) would imply strong regularity (see step 6 of Section 2.12.4).  Large
values would imply substantial fluctuation or irregularity
"""

MIN_BITS = 2

class ApproximateEntropy:

    def __init__(self, bits, p_threshold = 0.01):

        self.bits = bits
        self.p_threshold = p_threshold
        self.n = len(bits)

        self.m = None           # length of each window
        self.phi_m = []
        self.ApEn = None        # approximate entropy
        self.chisq = None       # test statistic
        self.p = None           # p value
        self.success = None
        self.test_run = False

        self.check_enough_bits()

        if self.enough_bits: self.run_test()

    def check_enough_bits(self):
        self.enough_bits = MIN_BITS <= self.n

    def run_test(self):
        self.m = max(min(int(np.floor(np.log2(self.n)))-6, 3), 2)
        # self.m = 10           # for testing vs NIST values
        self.phi_m = [self.get_phi(m) for m in range(self.m, self.m+2)]
        self.ApEn = self.phi_m[0] - self.phi_m[1]
        self.chisq = 2*self.n*(np.log(2) - self.ApEn)
        self.p = sp.gammaincc(2**(self.m-1),(self.chisq/2.0))
        self.success = (self.p >= 0.01)
        self.test_run = True

    def compare_ints(block_as_int, test_as_int):
        for test_as_int in range(len(counts)):
            print(test_as_int)
            if block_as_int == test_as_int:

                counts[test_as_int] += 1
                return counts

    def get_phi(self, m):
        padded_bits = self.bits+self.bits[0:m-1]      #  augment the sequence with the m-1 initial bits
        counts = [0 for i in range(2**m)]
        for bdx in range(self.n):
            block = padded_bits[bdx:bdx+m]
            block_as_int = int(''.join([str(i) for i in block]), 2)
            counts[block_as_int] += 1
        Ci = [i/self.n for i in counts]
        phi_m = 0
        for i in Ci:
            if i > 0: phi_m += i*np.log(i)
        return phi_m

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import utils

    bits = utils.e[:1000000]
    # bits = utils.from_string('0100110101')
    AE = ApproximateEntropy(bits)
    print(AE.p) # p = 0.700073
