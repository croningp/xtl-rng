import numpy as np
import scipy.special as sp

"""
The focus of the test is the rank of disjoint sub-matrices of the entire sequence.  The purpose of this test is
to check for linear dependence among fixed length substrings of the original sequence.

Note that large values of chisq (and hence, small P-values) would have indicated a deviation of the
rank distribution from that corresponding to a random sequence.
"""

MIN_BITS = 152

class Rank:

    def __init__(self, bits, p_threshold = 0.01):

        self.bits = bits
        self.p_threshold = p_threshold
        self.n = len(bits)

        self.M = None           # rows of matrix
        self.Q = None           # columns of matrix
        self.N = None           # number of matrices
        self.probs = []         # theoretical ratios of frequencies
        self.freqs = []         # observed frequencies

        self.chisq = None
        self.p = None
        self.success = None

        self.test_run = False
        self.check_enough_bits()

        if self.enough_bits: self.run_test()

    def check_enough_bits(self):
        self.enough_bits = MIN_BITS <= self.n

    def run_test(self):
        self.calculate_variables()

        blocks = [self.bits[i*self.M*self.Q:(i+1)*self.M*self.Q] for i in range(self.N)]
        matrices = [self.matrix_from_bits(block) for block in blocks]

        self.ranks = [self.rank(matrix) for matrix in matrices]
        self.freqs = self.get_rank_frequencies()

        self.chisq = sum([(f-(t*self.N))**2/(t*self.N) for f, t in zip(self.freqs, self.probs)])

        self.p = np.e **(-self.chisq/2.0)
        self.success = (self.p >= self.p_threshold)
        self.test_run = True

    def calculate_variables(self):

        self.M = self.Q = 32
        self.N = int(np.floor(self.n/(self.M*self.Q)))


        FR_prob = self.product(r=self.M)
        FRM1_prob = self.product(r=self.M-1)
        LR_prob = 1.0 - (FR_prob + FRM1_prob)

        self.probs = [FR_prob, FRM1_prob, LR_prob]

    def product(self, r):
        product = 1.0
        for i in range(int(r)):
            upper1 = (1.0 - (2.0**(i-self.Q)))
            upper2 = (1.0 - (2.0**(i-self.M)))
            lower = 1-(2.0**(i-r))
            product = product * ((upper1*upper2)/lower)
        return product * (2.0**((r*(self.Q+self.M-r)) - (self.M*self.Q)))

    def get_rank_frequencies(self):
        FM  = sum([1 for r in self.ranks if r == self.M])
        FMM  = sum([1 for r in self.ranks if r == self.M-1])
        remainder = len(self.ranks)-(FM+FMM)

        return [FM, FMM, remainder]

    def matrix_from_bits(self, block):

        m = [block[rownum*self.M:(rownum+1)*self.M] for rownum in range(self.Q)][:]
        return m

    def rank(self,matrix):
        lm = self.row_echelon(matrix)
        rank = 0
        for i in range(self.Q):
            nonzero = True if sum([bit for bit in lm[i]]) else False
            if nonzero: rank += 1
        return rank

    def row_echelon(self, matrix):
        lm = matrix
        psc, psr = 0, 0
        for i in range(self.Q):
            found = False
            for k in range(psr,self.Q):
                if lm[k][psc] == 1:
                    found = True
                    pr = k
                    break
            if found:
                if pr != psr: lm[pr],lm[psr] = lm[psr],lm[pr]
                for j in range(psr+1,self.Q):
                    if lm[j][psc]==1: lm[j] = [x ^ y for x,y in zip(lm[psr],lm[j])]
                psr += 1
            psc += 1

        return lm

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import utils

    bits = utils.e[:1000000]
    R = Rank(bits)
    print(R.p) # p = 0.306156
