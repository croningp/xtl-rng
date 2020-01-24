import numpy as np
import scipy.special as sp

"""
The focus of this test is the number of cycles having exactly K visits in a cumulative sum random walk.
The cumulative sum random walk is derived from partial sums after the (0,1) sequence is transferred to
the appropriate (-1, +1) sequence.  A cycle of a random walk consists of a sequence of steps of unit length
taken at random that begin at and return to the origin. The purpose of this test is to determine if the
number of visits to a particular state within a cycle deviates from what one would expect for a random
sequence. This test is actually a series of eight tests (and conclusions), one test and conclusion for each of
the states: -4, -3, -2, -1 and +1, +2, +3, +4.

Note that if chisq were too large, then the sequence would have displayed a deviation from the
theoretical distribution for a given state across all cycles.
"""

MIN_BITS = 5

class RandomExcursions:

    def __init__(self, bits, p_threshold = 0.01):

        self.bits = bits
        self.p_threshold = p_threshold
        self.n = len(bits)

        self.k = None
        self.cycles = []
        self.J = None
        self.chisq_list = [0 for i in range(8)]
        self.p_list = [None for i in range(8)]
        self.p = None           # p value
        self.success = None
        self.test_run = False

        self.check_enough_bits()

        if self.enough_bits: self.run_test()

    def check_enough_bits(self):
        self.enough_bits = MIN_BITS <= self.n


    def run_test(self):
        self.k = 5
        self.number_of_states = 8

        self.normed_bits = [bit*2-1 for bit in self.bits]   # conversion of every 0 to -1
        self.cusum = self.get_cusum()
        self.cusum_padded = [0] + self.cusum + [0]
        self.J = self.cusum_padded[1:].count(0)
        if self.J < 500:
            self.enough_bits = False
            return
        self.cycles = self.get_cycles()
        self.count_occurances()
        self.chisq_list = [0 for i in range(8)]

        for idx, state in enumerate(self.v):
            x = abs(idx-4) if idx < 4 else idx-3
            for jdx, k in enumerate(state):
                self.chisq_list[idx] += (k-self.J*self.ref_probs[x-1][jdx])**2/(self.J*self.ref_probs[x-1][jdx])
        self.p_list = [sp.gammaincc(5/2, chisq/2) for chisq in self.chisq_list]

        self.p = self.p_list[0]
        self.success = (self.p >= self.p_threshold)
        self.test_run = True

    def get_cusum(self):
        pos = 0
        cusum = []
        for bit in self.normed_bits:
            pos += bit
            cusum.append(pos)
        return cusum

    def get_cycles(self):
        pos = 1
        cycles = []
        for cycle in range(self.J):
            this_cycle = [0]
            for bit in self.cusum_padded[pos:]:
                this_cycle.append(bit)
                pos += 1
                if bit == 0:
                    break
            cycles.append(this_cycle)
        return cycles


    def count_occurances(self):

        occurances  = []
        for idx, cycle in enumerate(self.cycles):

            states = [0 for i in range(self.number_of_states)]
            for state in cycle[1:-1]:
                if state >= -4 and state <= 4: #(min(state, 4),-4)

                    if state < 0 :
                        states[state+4] += 1
                    else:
                        states[state+3] += 1
            occurances.append(states)

        self.occurances = np.asarray(occurances).T

        self.v = [[0 for k in range(6)] for s in range(8)]
        for idx, state in enumerate(self.occurances):

            for cycle in state:
                cycle = min(5, cycle)
                self.v[idx][cycle] += 1


    @property
    def ref_probs(self):
        return [[0.5     ,0.25   ,0.125  ,0.0625  ,0.0312 ,0.0312],
              [0.75    ,0.0625 ,0.0469 ,0.0352  ,0.0264 ,0.0791],
              [0.8333  ,0.0278 ,0.0231 ,0.0193  ,0.0161 ,0.0804],
              [0.875   ,0.0156 ,0.0137 ,0.012   ,0.0105 ,0.0733],
              [0.9     ,0.01   ,0.009  ,0.0081  ,0.0073 ,0.0656],
              [0.9167  ,0.0069 ,0.0064 ,0.0058  ,0.0053 ,0.0588],
              [0.9286  ,0.0051 ,0.0047 ,0.0044  ,0.0041 ,0.0531]]

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import utils

    bits = utils.e[:1000000]
    # bits = utils.from_string('0110110101')
    RE = RandomExcursions(bits)
    print(RE.p_list) # p = 0.786868
