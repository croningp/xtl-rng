import numpy as np
from scipy.stats import norm

"""
The focus of this test is the maximal excursion (from zero) of the random walk defined by the cumulative
sum of adjusted (-1, +1) digits in the sequence.  The purpose of the test is to determine whether the
cumulative sum of the partial sequences occurring in the tested sequence is too large or too small relative
to the expected behavior of that cumulative sum for random sequences.  This cumulative sum may be
considered as a random walk.  For a random sequence, the excursions of the random walk should be near
zero.  For certain types of non-random sequences, the excursions of this random walk from zero will be
large.

Note that when mode = 0, large values of this statistic indicate that there are either “too many ones” or
“too many zeros” at the early stages of the sequence; when mode = 1, large values of this statistic indicate
that there are either “too many ones” or “too many zeros” at the late stages.  Small values of the statistic
would indicate that ones and zeros are intermixed too evenly.
"""

MIN_BITS = 2

class CumulativeSums:

    def __init__(self, bits, p_threshold=0.01):
        self.bits = bits
        self.p_threshold = p_threshold
        self.n = len(bits)

        self.max_f = None       # max cusum in forward direction
        self.max_b = None       # max cusum in reverse direction
        self.p_f = None         # p-value in forward direction
        self.p_b = None         # p-value in reverse direction
        self.p = None           # p-value for test
        self.success = None

        self.test_run = False
        self.check_enough_bits()

        if self.enough_bits: self.run_test()

    def check_enough_bits(self):
        self.enough_bits = MIN_BITS <= self.n

    def run_test(self):
        self.normed_bits = [bit*2-1 for bit in self.bits]   # conversion of every 0 to -1

        self.max_f = max(self.get_psums(self.normed_bits))
        self.max_b = max(self.get_psums(reversed(self.normed_bits)))
        self.p_f = self.calc_p(self.max_f)
        self.p_b = self.calc_p(self.max_b)
        self.p = self.p_b
        self.success = self.p >= self.p_threshold
        self.test_run = True

    def calc_p(self, z):

        k0 = int(np.floor((self.n/z-1)/4))
        k1 = int(np.floor((1-self.n/z)/4))
        k2 = int(np.floor((-self.n/z-3)/4))

        sum1 = self.calc_part_sum(k1, k0, z, 1, -1)
        sum2 = self.calc_part_sum(k2, k0, z, 3, 1)

        p = 1 - sum1 + sum2
        return p

    def get_psums(self, bits):
        pos = 0
        psums = []
        for bit in bits:
            pos += bit
            psums.append(pos)
        psums = [abs(s) for s in psums]
        return psums

    def calc_part_sum(self, k_low, k_high, z, m1, m2):
        part_sum = 0
        for k in range(k_low, k_high+1):
            cdf1 = norm.cdf((4*k+m1)*z/np.sqrt(self.n))
            cdf2 = norm.cdf((4*k+m2)*z/np.sqrt(self.n))
            part_sum += (cdf1- cdf2)
        return part_sum
if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import utils

    bits = utils.e[:1000000]
    C = CumulativeSums(bits)
    print(C.p) # p = 0.669887
