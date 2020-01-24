import numpy as np
import scipy.special as sp

"""
The focus of the test is the proportion of zeroes and ones for the entire sequence.  The purpose of this test
is to determine whether the number of ones and zeros in a sequence are approximately the same as would
be expected for a truly random sequence.  The test assesses the closeness of the fraction of ones to ½, that
is, the number of ones and zeroes in a sequence should be about the same.  All subsequent tests depend on
the passing of this test.

Note that if the P-value were small (< 0.01), then this would be caused by s or being large.
Large positive values of s are indicative of too many ones, and large negative values of s are indicative
of too many zeros.
"""

MIN_BITS = 100

class Monobit:

    def __init__(self, bits, p_threshold = 0.01):

        self.bits = bits
        self.p_threshold = p_threshold
        self.n = len(bits)

        self.ones = None
        self.zeroes = None
        self.diff = None        # difference between the number of 0s and 1s
        self.absdiff = None     # absolute value of the difference
        self.excess = None      # which bit is in excess, 0s or 1s
        self.s = None           # s_obs is the test statistic
        self.p = None           # p_value = erfc(s_obs / sqrt(2))
        self.success = None

        self.test_run = False
        self.check_enough_bits()

        if self.enough_bits: self.run_test()

    def check_enough_bits(self):
        self.enough_bits = MIN_BITS <= self.n

    def run_test(self):
        self.ones = sum(self.bits)
        self.zeroes = len(self.bits) - sum(self.bits)

        self.diff = self.ones-self.zeroes
        self.absdiff = abs(self.diff)
        self.excess = 1 if self.diff == self.absdiff else 0

        self.s = self.absdiff/np.sqrt(self.n)
        self.p = sp.erfc(self.s/np.sqrt(2.0))
        self.success = self.p >= self.p_threshold
        self.test_run = True


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import utils
    bits = utils.e[:1000000]
    M = Monobit(bits)
    print(M.p) # p = 0.953749
