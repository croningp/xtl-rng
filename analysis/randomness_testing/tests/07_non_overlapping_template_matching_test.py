import sys
import numpy as np
import scipy.special as sp
import math

"""
The focus of this test is the number of occurrences of  pre-specified target strings.  The purpose of this
test is to detect generators that produce too many occurrences of a given non-periodic (aperiodic) pattern.
For this test and for the Overlapping Template Matching test of Section 2.8, an m-bit window is used to
search for a specific m-bit pattern. If the pattern is not found, the window slides one bit position. If the
pattern is found, the window is reset to the bit after the found pattern, and the search resumes.

If the P-value is very small (< 0.01), then the sequence has irregular occurrences of the possible template
patterns.
"""
MIN_BITS = 10

class NonOverlappingTemplateMatching:

    def __init__(self, bits, p_threshold = 0.01):

        self.bits = bits
        self.p_threshold = p_threshold
        self.n = len(bits)

        self.B = []             # template to be used
        self.m = None           # length of template in bits
        self.N = None           # number of blocks
        self.M = None           # length of each block
        self.W = []             # number of times template appears in each block
        self.mu = None          # theoretical mean for W
        self.sigma = None       # theoretical variance for W
        self.chisq = None       # test statistic
        self.p = None           # p value
        self.success = None
        self.test_run = False   # test was run successfully

        self.check_enough_bits()

        if self.enough_bits: self.run_test()

    def check_enough_bits(self):
        self.enough_bits = MIN_BITS <= self.n

    def run_test(self):

        self.B = [0,0,0,0,0,0,0,0,1]
        self.m = len(self.B)

        self.N = 8
        self.M = int(np.floor(len(self.bits)/self.N))
        blocks = [self.bits[i*self.M:(i+1)*self.M] for i in range(self.N)]

        self.W = [self.get_count(block) for block in blocks]

        self.mu = float(self.M-self.m+1)/float(2**self.m)
        self.sigma = self.M*(1/2**self.m-(2*self.m-1)/2**(2*self.m))

        self.chisq = sum([(Wj-self.mu)**2/self.sigma for Wj in self.W])
        self.p = sp.gammaincc(self.N/2, self.chisq/2)
        self.success = (self.p >= self.p_threshold)

        self.test_run = True

    def get_count(self, block):
        position, count = 0, 0
        while position < (self.M-self.m):
            if block[position:position+self.m] == self.B:
                position += self.m-1
                count += 1
            position += 1
        return count

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import utils

    bits = utils.e[:1000000]
    NOTM = NonOverlappingTemplateMatching(bits)
    print(NOTM.p) # p = 0.078790
