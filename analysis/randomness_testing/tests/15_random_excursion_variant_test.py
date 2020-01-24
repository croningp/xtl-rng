import numpy as np
import scipy.special as sp

"""
The focus of this test is the total number of times that a particular state is visited (i.e., occurs) in a
cumulative sum random walk. The purpose of this test is to detect deviations from the expected number
of visits to various states in the random walk. This test is actually a series of eighteen tests (and
conclusions), one test and conclusion for each of the states: -9, -8, ..., -1 and +1, +2, ..., +9.

"""

MIN_BITS = 0

class RandomExcursionsVariant:

    def __init__(self, bits, p_threshold = 0.01):

        self.bits = bits
        self.p_threshold = p_threshold
        self.n = len(bits)

        self.J = None
        self.counts = []
        self.p_list = []

        self.p = None           # p value
        self.success = None
        self.test_run = False

        self.check_enough_bits()

        if self.enough_bits: self.run_test()

    def check_enough_bits(self):
        self.enough_bits = MIN_BITS <= self.n

    def run_test(self):

        self.normed_bits = [bit*2-1 for bit in self.bits]   # conversion of every 0 to -1
        self.psums = self.get_psums()
        self.psums_padded = [0] + self.psums + [0]
        self.J = sum([1 for v in self.psums_padded[1:] if v == 0])
        self.get_counts()
        self.p_list = [self.calc_p(idx) for idx in range(-9,10)]
        self.p_list.pop(9)

        self.p = self.p_list[8]
        self.success = (self.p >= self.p_threshold)
        self.test_run = True

    def get_counts(self):
        self.counts = [0 for i in range(19)]
        for v in self.psums_padded:
            if abs(v) < 10: self.counts[v+9] += 1

    def get_psums(self):
        pos = 0
        psums = []
        for bit in self.normed_bits:
            pos += bit
            psums.append(pos)
        return psums

    def calc_p(self,idx):
        if idx == 0: return 1
        numerator = abs(self.counts[idx+9]-self.J)
        denominator = np.sqrt(2.0*self.J*(4.0*abs(idx)-2.0))
        p = sp.erfc(numerator / denominator)
        return p

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import utils

    bits = utils.e[:1000000]
    REV = RandomExcursionsVariant(bits)
    print(REV.p_list[8]) # p = 0.826009
