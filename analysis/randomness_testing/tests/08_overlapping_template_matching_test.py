import sys
import numpy as np
import scipy.special as sp

"""
The focus of the Overlapping Template Matching test is the number of occurrences of pre-specified target
strings. Both this test and the Non-overlapping Template Matching test use an m-bit window to search for
a specific m-bit pattern. As with the Non-overlapping Template Matching test, if the pattern is not
found, the window slides one bit position. The difference between this test and the test in Section 2.7 is that
when the pattern is found, the window slides only one bit before resuming the search.

Note that for the 2-bit template (B = 11), if the entire sequence had too many 2-bit runs of ones, then:
1) ν5 would have been too large, 2) the test statistic would be too large, 3) the P-value would have been
small (< 0.01) and 4) a conclusion of non-randomness would have resulted.
"""



MIN_BITS = 100

class OverlappingTemplateMatching:

    def __init__(self, bits, p_threshold = 0.01):

        """ Note: This test expects at least 1000000 bits """

        self.bits = bits
        self.p_threshold = p_threshold
        self.n = len(self.bits)

        self.m = None           # length of bits in the template
        self.B = []             # the m-bit template to be matched
        self.K = None           # Number of degrees of freedom
        self.M = None           # length of each substring in bits
        self.N = None           # number of substring blocks
        self.pi = []
        self.lmbda = None
        self.eta = None
        self.piqty = []
        self.probs = []

        self.v = []             # observed frequencies

        self.chisq = None       # test statistic
        self.p = None           # p value
        self.success = None
        self.test_run = False

        self.check_enough_bits()

        if self.enough_bits: self.run_test()

    def check_enough_bits(self):
        self.enough_bits = MIN_BITS <= self.n

    def run_test(self):
        self.m = 9
        self.B = [1 for x in range(self.m)]
        self.N = 968
        self.K = 5
        self.M = 1032
        self.pi = [0.364091, 0.185659, 0.139381, 0.100571, 0.0704323, 0.139865]

        self.lmbda = (self.M-self.m+1)/2**self.m
        self.eta = self.lmbda / 2

        blocks = [self.bits[i*self.M:(i+1)*self.M] for i in range(self.N)]
        self.v = [0 for x in range(self.K+1)]
        for block in blocks:
            count = 0
            for idx in range(len(block)):
                window = block[idx:idx+self.m]
                if window == self.B:
                    count += 1
            count = min(count, self.K)
            self.v[count] += 1
        self.probs = [self.compute_prob(i) for i in range(self.K)]
        self.probs.append(1-sum(self.probs))

        self.chisq = self.calc_chisq()

        self.p = sp.gammaincc(5/2, self.chisq/2)
        self.success = (self.p >= 0.01)
        self.test_run = True

    def calc_chisq(self):
        chisq = 0
        for i in range(self.K+1):
            numerator = (self.v[i]-self.N*self.probs[i])**2
            denominator = self.N*self.probs[i]
            chisq += numerator / denominator
        return chisq


    def compute_prob(self, i):
        if i == 0: return np.exp(-self.eta)
        prob = 0.0
        for l in range(1,i+1):
            prob += np.exp(-self.eta-i*np.log(2)+l*np.log(self.eta)
                    -self.lgamma(l+1)+self.lgamma(i)-self.lgamma(l)-self.lgamma(i-l+1))
        return prob

    def lgamma(self, i):
        return np.log(sp.gamma(i))

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import utils

    bits = utils.e[:1000000]
    # bits = utils.pseudo(10000)
    # bits = utils.from_string('10111011110010110100011100101110111110000101101001')
    OTM = OverlappingTemplateMatching(bits)
    print(OTM.p) # p = 0.110434
