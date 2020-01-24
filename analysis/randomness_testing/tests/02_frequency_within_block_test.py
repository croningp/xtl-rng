import numpy as np
import scipy.special as sp

"""
The focus of the test is the proportion of ones within M-bit blocks. The purpose of this test is to determine
whether the frequency of ones in an M-bit block is approximately M/2, as would be expected under an
assumption of randomness.  For block size M=1, this test degenerates to test 1, the Frequency (Monobit)
test.

Note that small P-values (< 0.01) would have indicated a large deviation from the equal proportion of
ones and zeros in at least one of the blocks.
"""


MIN_BITS = 100
MIN_BLOCK_LENGTH = 20
MAX_NUMBER_OF_BLOCKS = 99

class BlockFrequency:

    def __init__(self, bits, p_threshold=0.01):
        self.bits = bits
        self.p_threshold = p_threshold

        self.n = len(bits)
        self.M = None               # length of blocks
        self.N = None               # number of blocks
        self.chisq = None           # chisq is the test statistic

        self.p = None               # p_value = erfc(s_obs / sqrt(2))
        self.success = None
        self.test_run = False

        self.check_enough_bits()

        if self.enough_bits: self.run_test()

    def check_enough_bits(self):
        self.enough_bits = MIN_BITS <= self.n

    def run_test(self):
        self.N = min(MAX_NUMBER_OF_BLOCKS, int(np.floor(self.n/MIN_BLOCK_LENGTH)))
        self.M = int(np.floor(self.n/self.N))
        self.M = 128                            # for testing vs NIST values
        self.N = int(np.floor(self.n/self.M))   # for testing vs NIST values
        blocks = [self.bits[i*self.M:(i+1)*self.M] for i in range(self.N)]
        proportions = [sum(block)/len(block) for block in blocks]

        self.chisq = 4*self.M*sum([(prop-0.5)**2 for prop in proportions])
        self.p = sp.gammaincc(self.N/2, self.chisq/2)
        self.success = self.p >= self.p_threshold
        self.test_run = True

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import utils

    bits = utils.e[:1000000]
    BF = BlockFrequency(bits)
    print(BF.p) # p = 0.211072 (when m = 128)
