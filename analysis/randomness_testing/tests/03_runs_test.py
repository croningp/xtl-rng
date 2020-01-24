import numpy as np
import scipy.special as sp

"""
The focus of this test is the total number of runs in the sequence, where a run is an uninterrupted sequence
of identical bits.  A run of length k consists of exactly k identical bits and is bounded before and after with
a bit of the opposite value. The purpose of the runs test is to determine whether the number of runs of
ones and zeros of various lengths is as expected for a random sequence.  In particular, this test determines
whether the oscillation between such zeros and ones is too fast or too slow.

Note that a large value for v would have indicated an oscillation in the string which is too fast; a
small value would have indicated that the oscillation is too slow.  (An oscillation is considered to be a
change from a one to a zero or vice versa.)  A fast oscillation occurs when there are a lot of changes, e.g.,
010101010 oscillates with every bit.  A stream with a slow oscillation has fewer runs than would be
expected in a random sequence, e.g., a sequence containing 100 ones, followed by 73 zeroes, followed by
127 ones (a total of 300 bits) would have only three runs, whereas 150 runs would be expected.
"""

MIN_BITS = 100

class Runs:

    def __init__(self, bits, p_threshold=0.01):
        self.bits = bits
        self.p_threshold = p_threshold

        self.n = len(bits)
        self.v = None           # v_obs is the test statistic

        self.p = None           # p_value = erfc(abs(v_obs-numerator)/denominator)
        self.success = None
        self.test_run = False

        self.check_enough_bits()

        if self.enough_bits: self.check_monobit_passed()
        if self.monobit_passed: self.run_test()

    def check_enough_bits(self):
        self.enough_bits = MIN_BITS <= self.n

    def check_monobit_passed(self):
        self.prop = sum(self.bits)/len(self.bits)
        self.tau = 2/np.sqrt(self.n)
        self.monobit_passed = abs(self.prop-0.5) <= self.tau

    def run_test(self):
        self.v = sum([1 for i in range(self.n-1) if self.bits[i] != self.bits[i+1]]) + 1
        factor = 2*self.prop*(1-self.prop)
        self.p = sp.erfc(abs(self.v-self.n*factor)/(np.sqrt(2*self.n)*factor))
        self.success = self.p >= self.p_threshold
        self.test_run = True

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import utils

    bits = utils.e[:1000000]
    R = Runs(bits)
    print(R.p) # p = 0.561917
