import os
import sys

import numpy as np
import scipy.special as sp


"""
The focus of this test is the length of a linear feedback shift register (LFSR).  The purpose of this test is to
determine whether or not the sequence is complex enough to be considered random.  Random sequences
are characterized by longer LFSRs.  An LFSR that is too short implies non-randomness

If the P-value were < 0.01, this would have indicated that the observed frequency counts of Ti stored in
the νI bins varied from the expected values; it is expected that the distribution of the frequency of the
Ti (in the νI bins) should be proportional to the computed pi_i.

"""

MIN_BITS = 1

class LinearComplexity:

    def __init__(self, bits, p_threshold = 0.01):

        self.bits = bits
        self.p_threshold = p_threshold
        self.n = len(bits)

        self.K = None           # degrees of freedom
        self.M = None           # length of block in bits
        self.N = None           # number of blocks

        self.LC = []            # linear complexities of blocks
        self.mu = None          # theoretical mean
        self.T = []             # difference from mean
        self.v = [0,0,0,0,0,0,0]


        self.p = None           # p value
        self.success = None
        self.test_run = False

        self.check_enough_bits()

        if self.enough_bits: self.run_test()


    def check_enough_bits(self):
        self.enough_bits = MIN_BITS <= self.n

    def run_test(self):
        self.M = 500
        self.K = 6
        self.N = int(np.floor(self.n/self.M))

        blocks = [self.bits[(i*self.M):((i+1)*self.M)] for i in range(self.N)]

        for idx, block in enumerate(blocks):
            self.LC.append(self.berelekamp_massey(block))

        self.mu = float(self.M)/2+((((-1)**(self.M+1))+9))/36-((self.M/3)+(2/9))/(2**self.M)
        self.T = [(-1)**self.M*(self.LC[i]-self.mu)+(2/9) for i in range(self.N)]
        for t in self.T:
            self.v[self.get_idx(t)] += 1

        self.chisq = self.get_chisq()
        self.p = sp.gammaincc(self.K/2,self.chisq/2)
        self.success = (self.p >= self.p_threshold)
        self.test_run = True

    def get_chisq(self):
        chisq = 0
        for i in range(self.K+1):
            numerator = (self.v[i] - self.N*self.pi[i])**2
            denominator = self.N*self.pi[i]
            chisq += numerator / denominator
        return chisq


    def get_idx(self, t):
        if t <= -2.5: return 0
        if t <= -1.5: return 1
        if t <= -0.5: return 2
        if t <= 0.5: return 3
        if t <= 1.5: return 4
        if t <= 2.5: return 5
        return 6

    @property
    def pi(self):
        return [0.010417,0.03125,0.125,0.5,0.25,0.0625,0.020833]

    @property
    def data(self):
        return {'n':self.n, 'K':self.K, 'M':self.M, 'N':self.N, 'LC': self.LC,
                'mu': self.mu, 'T':self.T, 'v':self.v, 'chisq':self.chisq,
                'p':self.p, 'success':self.success, 'run':self.test_run}

    def get_linear_complexities(self):
        pass

    def berelekamp_massey(self, bits):
        n=len(bits)
        b = [0 for x in bits]  #initialize b and c arrays
        c = [0 for x in bits]
        b[0] = 1
        c[0] = 1

        L = 0
        m = -1
        N = 0
        while (N < self.M):

            d = bits[N]
            for i in range(1,L+1):
                d = d ^ (c[i] & bits[N-i])
            if (d != 0):
                t = c[:]
                for i in range(0,self.M-N+m):
                    c[N-m+i] = c[N-m+i] ^ b[i]
                if (L <= (N/2)):
                    L = N + 1 - L
                    m = N
                    b = t
            N = N +1
        return L


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import utils

    bits = utils.e[:1000000]
    # bits = utils.from_string('1101011110001')
    LC = LinearComplexity(bits)
    print(LC.p) # p = 0.282568
