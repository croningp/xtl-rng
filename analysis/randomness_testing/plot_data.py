import os
import sys
import inspect
import csv
import numpy as np

import utils

HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PROJECT_DIR =os.path.dirname(os.path.dirname(HERE_PATH))

from run_tests import Tester

class Plotter:

    def __init__(self, testnames):

        self.testnames = testnames
        self.data = None
        self.certainty_threshold = 0.01

        self.p_values = {}
        self.hists = {}
        self.pass_rates = {}
        self.uniformities = {}

    def new(self, input_file, number_of_blocks=None, block_size=None):

        self.string_filename = input_file
        self.string = utils.from_file(self.string_filename)
        self.tester = Tester(testnames)
        self.tester.run_all_tests(self.string, number_of_blocks, block_size)

        for test in self.testnames:
            self.p_values[test]     = [r.p for r in self.tester.results[test]]

    def process_data(self):
        for test in self.testnames:

            self.pass_rates[test]   = self.calc_pass_rates(self.p_values[test])
            self.hists[test]        = self.calc_histogram(self.p_values[test])
            self.uniformities[test] = self.calc_uniformity(self.hists[test])



    def plot_first_order(self, testnames=None, compound=''):
        if testnames == None:
            testnames = self.testnames
        xs = np.linspace(0.05, 0.95, 10)
        self.fig, axs = plt.subplots(len(testnames), sharex='col', figsize=(6.3,9))
        title1 = 'original string length: {}\nSubstring length: {} sample size: {}\n\n'.format(self.string_size, self.block_size, self.number_of_blocks)
        title2 = 'Histograms of p-values generated by statistical tests of the NIST\npackage for a binary string generated by crystallization of {}'.format(compound)
        self.fig.suptitle(title1+title2)

        for idx, test in enumerate(testnames):

            axs[idx].bar(xs, self.hists[test], width=0.1, edgecolor='k')

            ymax = int(max(self.hists[test]) *1.5)
            axs[idx].set_ylim(ymin=0, ymax = ymax)
            text_height = (ymax*0.8+max(self.hists[test]))/2
            if test != 'NonOverlappingTemplateMatching':
                axs[idx].text(0,text_height, test)
            else:
                axs[idx].text(0,text_height, test, fontsize=8)

            axs[idx].text(0.45,text_height,'{}'.format(r'$p_{uniform}$'+'    ={0:.4f}'.format(self.uniformities[test])))
            axs[idx].text(0.75, text_height,'pass rate= {0:.4f}'.format(self.pass_rates[test]))
            axs[idx].minorticks_on()

            if idx == int(len(testnames)/2) -1:
                axs[idx].set_ylabel('count', rotation=90)

        axs[idx].set_xlabel('p-value range')
        plt.savefig('{} - first order.svg'.format(compound))
        plt.show()

    def plot_second_order(self, testnames=None, compound =''):
        if testnames == None:
            testnames = self.testnames
        title = 'Pass rates for statistical tests of the NIST package\nfor a binary string generated by crystallization of {}'.format(compound)
        xs = []
        ys = []
        plt.title(title)
        for test in testnames:
            xs.append(test)
            ys.append(self.pass_rates[test])
        plt.bar(xs, ys,width=0.8,edgecolor='k')

        plt.hlines([1], [-1],[len(testnames)],  linestyle='-.')
        plt.hlines([self.pass_min], [-1],[len(testnames)],  linestyle='-')
        plt.hlines([self.p_hat], [-1],[len(testnames)],  linestyle='--')
        plt.hlines([self.pass_max], [-1],[len(testnames)],  linestyle='-')
        plt.xticks(rotation=30, ha='right')
        plt.xlim(-1,len(testnames))
        y_max = (self.pass_max + self.p_hat) / 2
        y_min = (self.pass_min+self.p_hat)-1
        plt.ylim(y_min, y_max)
        plt.ylabel('Pass rate')
        plt.savefig('{} - second order.svg'.format(compound))
        plt.show()

    def plot_third_order(self, testnames=None, compound= ''):
        if testnames == None:
            testnames = self.testnames
        title = 'Uniformity of p-values for histograms of tests of the NIST\npackage for a binary string generated by crystallization of {}'.format(compound)
        xs = []
        ys = []
        plt.title(title)
        for test in testnames:
            xs.append(test)
            ys.append(self.uniformities[test])
        plt.bar(xs, ys, width=0.8, edgecolor='k')
        plt.xticks(rotation=30, ha='right')
        plt.xlim(-1,len(testnames))
        plt.ylabel('Uniformity p-value')
        plt.savefig('{} - third order.svg'.format(compound))
        plt.show()



    def calc_histogram(self, p_values):
        return np.histogram(p_values, 10, (0,1))[0]

    def calc_uniformity(self, hist):
        chi_sq = sum([(i-(sum(hist)/10))**2/(sum(hist)/10) for i in hist])
        p_value = 1-sp.gammainc(9/2, chi_sq/2)
        return p_value

    def calc_pass_rates(self, p_values):
        pass_rate =  sum([1 for p in p_values if p > 0.01])/len(p_values)

        self.p_hat = 1- self.certainty_threshold
        pass_window = 3*np.sqrt((self.p_hat*(1-self.p_hat)/len(p_values)))

        self.pass_min = self.p_hat - pass_window
        self.pass_max = self.p_hat + pass_window

        return pass_rate

    def save_p_values(self, filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['string size', self.tester.total_bits])
            writer.writerow(['block size', self.tester.block_size])
            writer.writerow(['number of blocks', self.tester.number_of_blocks])
            for test in self.testnames:
                row = [test] + self.p_values[test]
                writer.writerow(row)

    def load_p_values(self, filename):

        with open(filename, 'r') as f:
            reader = csv.reader(f)
            self.string_size = next(reader)[1]
            self.block_size = next(reader)[1]
            self.number_of_blocks = next(reader)[1]
            for row in reader:
                self.p_values[row[0]] = [float(i) for i in row[1:] if i != '']


    def savefig(self, output_file):
        self.fig.savefig(output_file)

TESTNAMES = [
            'Monobit',                            # fast  pass    pass
            'BlockFrequency',                     # fast  pass    pass
            'Runs',                               # fast  pass    pass
            'LongestRunOfOnes',                   # fast  pass    pass
            'Rank',                               # fast  pass    pass
            'DiscreteFourierTransform',           # fast  pass    pass
            'NonOverlappingTemplateMatching',     # fast  pass
            # 'OverlappingTemplateMatching',        # fast   pass
            'Universal',                          # fast  pass
            'LinearComplexity',                   # slow  pass
            'Serial',                             # fast  pass
            'ApproximateEntropy',                 # fast  pass    pass
            'CumulativeSums',                     # fast  pass    pass
            'RandomExcursions',                   # fast  pass
            'RandomExcursionsVariant'             # fast  pass
            ]

if __name__ == '__main__':

    testnames = TESTNAMES
    compound = 'CuSO4'

    filepath = os.path.join(PROJECT_DIR, 'results', 'numbers', compound)

    output_file = '{}.csv'.format(compound)
    p = Plotter(testnames)
    
    import platform
    if platform.system() == 'Linux':
        p.new(input_file, block_size=100000)
        p.save_p_values(output_file)
    else:

        import matplotlib.pyplot as plt
        import scipy.special as sp

        p.load_p_values(output_file)

        p.process_data()

        p.plot_first_order(testnames, compound)

        plt.clf()
        p.plot_second_order(testnames, compound)
        plt.clf()
        p.plot_third_order(testnames, compound)
        plt.clf()
