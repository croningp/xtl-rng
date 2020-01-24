import numpy as np
import scipy.special as sp

"""
The focus of the test is the longest run of ones within M-bit blocks. The purpose of this test is to
determine whether the length of the longest run of ones within the tested sequence is consistent with the
length of the longest run of ones that would be expected in a random sequence. Note that an irregularity in
the expected length of the longest run of ones implies that there is also an irregularity in the expected
length of the longest run of zeroes.  Therefore, only a test for ones is necessary.

Note that large values of chisq indicate that the tested sequence has clusters of ones.
"""

MIN_BITS = 128

class LongestRunOfOnes:

    def __init__(self, bits, p_threshold=0.01):
        self.bits = bits
        self.p_threshold = p_threshold

        self.n = len(bits)

        self.M = None               # length of blocks
        self.N = None               # number of blocks
        self.K = None               # another statistic
        self.bins = []              # bin range for histograms
        self.chisq = None           # chisq is the test statistic

        self.p = None
        self.success = None
        self.test_run = False

        self.check_enough_bits()

        if self.enough_bits: self.run_test()

    def check_enough_bits(self):
        self.enough_bits = MIN_BITS <= self.n

    def run_test(self):
        self.M, self.K, self.bins = self.calculate_variables()
        self.N = int(np.floor(self.n/self.M))
        blocks = [self.bits[i*self.M:(i+1)*self.M] for i in range(self.N)]
        longest_runs = [self.get_longest_run(block) for block in blocks]
        thresholded_runs = [min(max(self.bins[0], run), self.bins[-1]) for run in longest_runs]
        self.hist, bins = np.histogram(thresholded_runs, len(self.bins), range=(self.bins[0], self.bins[-1]+1))
        self.chisq = sum([(b-self.N*p)**2/(self.N*p) for b, p in zip(self.hist, self.probs)])
        self.p = sp.gammaincc(self.K/2, self.chisq/2)

        self.success = self.p >= self.p_threshold
        self.test_run = True

    def calculate_variables(self):
        if self.n < 6272: return 8,3, list(range(1,5))
        if self.n < 750000: return 128,5, list(range(4,10))
        return 10000,6, list(range(10,17))

    def get_longest_run(self, block):
        longest_run, this_run = 0, 0
        for bit in block:
            if bit == 1:
                this_run += 1
                if this_run > longest_run: longest_run = this_run
            else:
                this_run = 0
        return longest_run

    @property
    def probs(self):

        if self.M == 8: return [0.2148, 0.3672, 0.2305, 0.1875]
        if self.M == 128: return [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        return [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import utils

    bits = utils.e[:1000000]
    LROO = LongestRunOfOnes(bits)
    print(LROO.p) # p = 0.718945
