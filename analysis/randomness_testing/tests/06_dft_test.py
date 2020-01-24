import numpy as np
import scipy.special as sp

"""
The focus of this test is the peak heights in the Discrete Fourier Transform of the sequence.  The purpose
of this test is to detect periodic features (i.e., repetitive patterns that are near each other) in the tested
sequence that would indicate a deviation from the assumption of randomness. The intention is to detect
whether the number of peaks exceeding the 95 % threshold is significantly different than 5 %.

A d value that is too low would indicate that there were too few peaks (< 95%) below T, and too many
peaks (more than 5%) above T.
"""

MIN_BITS = 1000

class DiscreteFourierTransform:

    def __init__(self, bits, p_threshold = 0.01):

        self.bits = bits if len(bits) % 2 == 0 else bits[:-1]
        self.p_threshold = p_threshold
        self.n = len(self.bits)

        self.T = None           # peak height threshold value
        self.N0 = None          # theoretical number of peaks below threshold
        self.N1 = None          # observed number of peaks below threshold

        self.d = None           # test statistic
        self.p = None           # p value

        self.success = None
        self.test_run = False

        self.check_enough_bits()

        if self.enough_bits: self.run_test()

    def check_enough_bits(self):
        self.enough_bits = MIN_BITS <= self.n

    def run_test(self):

        normed_bits = np.array([bit*2-1 for bit in self.bits])

        peaks = abs(np.fft.fft(normed_bits)[:self.n//2])

        self.T = np.sqrt(np.log(1/0.05)*self.n) # Compute upper threshold
        self.N0 = 0.95*self.n/2.0
        self.N1 = sum([1 for peak in peaks if peak < self.T])

        self.d = (self.N1 - self.N0)/np.sqrt((self.n*0.95*0.05)/4) # Compute the P value
        self.p = sp.erfc(abs(self.d)/np.sqrt(2))

        self.success = (self.p >= 0.01)
        self.test_run = True

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import utils

    bits = utils.e[:1000000]
    DFT = DiscreteFourierTransform(bits)
    print(DFT.p) # p = 0.847187 
