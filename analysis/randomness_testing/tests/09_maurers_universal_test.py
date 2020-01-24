import numpy as np
import scipy.special as sp

"""
The focus of this test is the number of bits between matching patterns (a measure that is related to the
length of a compressed sequence).  The purpose of the test is to detect whether or not the sequence can be
significantly compressed without loss of information.  A significantly compressible sequence is
considered to be non-random.

If fn differs significantly from expectedValue(L), then the sequence is significantly compressible
"""


MIN_BITS = 5

class Universal:

    def __init__(self, bits, p_threshold = 0.01):

        self.bits = bits
        self.p_threshold = p_threshold
        self.n = len(bits)

        self.L = None           # length of each block
        self.M = None           # total number of blocks
        self.Q = None           # number of blocks in initialization sequence
        self.K = None           # blocks in test sequence
        self.sigma = None       # theoretical standard deviation
        self.T = []             # frequency table
        self.fn = None          # test statistic

        self.p = None           # p value
        self.success = None
        self.test_run = False

        self.check_enough_bits()

        if self.enough_bits: self.run_test()

    def check_enough_bits(self):
        self.enough_bits = MIN_BITS <= self.n

    def run_test(self):

        n = self.n
        bits = self.bits

        self.L = self.calc_L()
        self.M = int(np.floor(n/self.L))
        self.Q = 10*2**self.L
        self.K = self.M - self.Q
        self.eV = self.ev_table[self.L]
        self.var = self.var_table[self.L]
        self.c = 0.7-(0.8/self.L)+(4+32/self.L)*self.K**(-3/self.L)/15
        self.sigma = np.sqrt(self.var/self.K)*self.c

        blocks = [self.bits[i*self.L:(i+1)*self.L] for i in range(self.M)]

        self.init_segment = blocks[:self.Q]
        self.test_segment = blocks[self.Q:]

        self.init_table()
        self.fn = sum([self.get_distances(idx, test_bits) for idx, test_bits in enumerate(self.test_segment)])/self.K

        self.p = sp.erfc(abs((self.fn-self.eV) / (np.sqrt(2)*self.sigma)))
        self.success = (self.p >= self.p_threshold)

        self.test_run = True

    def calc_L(self):
        for idx, threshold_n in enumerate(self.ns):
            if self.n < threshold_n:
                 return idx+5
        return 4

    def init_table(self):
        self.T=[0 for x in range(2**self.L)]
        for idx, bits in enumerate(self.init_segment):
            self.T[self.bits2int(bits)] = idx+1

    def get_distances(self, idx, test_bits):

        occurance_idx = idx+self.Q+1
        previous_idx = self.bits2int(test_bits)
        dist = occurance_idx-self.T[previous_idx]
        self.T[previous_idx] = occurance_idx
        return np.log2(dist)

    def bits2int(self, bits):
        n = 0
        for bit in (bits):
            n = (n << 1) + bit
        return n


    @property
    def ns(self):
        return [387840, 904960,2068480,4654080,10342400,
                          22753280,49643520,107560960,
                          231669760,496435200,1059061760]
    @property
    def ev_table(self):
        return  [0,0.73264948,1.5374383,2.40160681,3.31122472,
                     4.25342659,5.2177052,6.1962507,7.1836656,
                     8.1764248,9.1723243,10.170032,11.168765,
                     12.168070,13.167693,14.167488,15.167379]
    @property
    def var_table(self):
        return [0,0.690,1.338,1.901,2.358,2.705,2.954,3.125, 3.419,
                     3.238,3.311,3.356,3.384,3.401,3.410,3.416, 3.421]

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import utils

    bits = utils.e[:1000000]
    u = Universal(bits)
    print(u.p) # p = 0.282568
