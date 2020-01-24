import os
import sys
import numpy as np

import scipy.special as sp

import matplotlib
import matplotlib.pyplot as plt

#in order to find where the tests are located
sys.path.append('sp800_22_tests')

def read_bits_from_ascii_file(filepath):
    # obtains all integers from an ascii file and puts them into a list
    # returns list of integers
    bitstring = ''
    for row in open(filepath, 'r').readlines():
        bitstring += row.rstrip('\n').strip()
    bitlist = [int(i) for i in bitstring]
    return bitlist


def read_binary_expansion_file(filepath):

    with open(filepath, 'r') as f:
        string = f.read()
    string = string.replace(' ', "")
    string = string.replace('\r', "")
    string = string.replace('\n', "")
    string = [int(i) for i in list(string)]
    
    return string

def get_chunks(bitlist, number_of_chunks=None, chunk_length=None):
    # partitions long bitstring into several small chunks of set length
    # returns list of list of integers
    
    # returns original string (in a list) if no parameters are set
    if not number_of_chunks and not chunk_length:
        number_of_chunks, chunk_length = 1, len(bits)
        
    # calculates chunk_length if only number_of_chunks set
    elif number_of_chunks and not chunk_length:
        chunk_length = int(len(bitlist) / number_of_chunks)
    
    # calculates number_of_chunks if only chunk_length set
    elif not number_of_chunks and chunk_length:
        number_of_chunks = int(len(bitlist) / chunk_length)
    
    # generates specified number of chunks with specified length
    chunks = [bitlist[idx*chunk_length:(idx+1)*chunk_length] for idx in range(number_of_chunks)]
    return chunks

def run_chunks(chunks):
    # assess each chunk in a loop
    # results are a dictionary with dict[testname] = [p-values] for all testnames
    chunk_results = list()

    print('Number of chunks: {}, size of chunks: {}'.format(len(chunks), len(chunks[0])))
    len_chunks = len(chunks)
    for idx, chunk in enumerate(chunks):
        chunk_results.append(run_testlist(idx, len_chunks, chunk))
    return chunk_results

def run_testlist(idx, len_chunks, bits):
    
    results = list()
    for testname in testlist:
        # print('Running chunk {} / {}. Test: {}                                                  \r'.format(idx, len_chunks, testname), end='\r', flush=True)      
        m = __import__ ("sp800_22_"+testname)
        func = getattr(m,testname) 
        (success,p,plist) = func(bits)

        summary_name = testname
        if success:
            summary_result = "PASS"
        else:
            summary_result = "FAIL"
        
        if p != None:
            summary_p = str(p)
            
        # if plist != None:
        #     summary_p = str(min(plist))
        
        results.append((summary_name,summary_p, summary_result))
    return results

def collate_results(data):
    # create results dictionary where dict[testname] = p-value
    res_dict = {}
    for test in testlist:
        res_dict[test] = []
    for chunk in data:       
        for test in chunk:
            res_dict[test[0]].append(float(test[1]))
    return res_dict 

def calc_stats(data):
    # generate 11 bin edges -> 10 bins between 0 and 1
    bins = np.linspace(0,1,11)
    stats = {}
    # create stats dictionary where dict[testname] = dictionary of test statistics for each testname
    for test in data:
        stats[test] = {}
    for test in data:
        hist, edges = np.histogram(data[test], bins)
        stats[test]['hist'] = hist
        stats[test]['hist axis'] = np.linspace(0.05, 0.95, 10)
        stats[test]['uniformity'] = calc_uniformity(hist)
        stats[test]['pass rate'] = sum([1 for p in data[test] if p > 0.01])/len(data[test])
        stats[test]['pass window'] = calc_passwindow(data[test])
       
    return stats

def calc_uniformity(hist):
    chi_sq = sum([(i-(sum(hist)/10))**2/(sum(hist)/10) for i in hist])
    p_value = 1-sp.gammainc(9/2, chi_sq/2)
    return p_value

def calc_passwindow(p_values):

        p_hat = 0.99
        pass_window = 3*np.sqrt((p_hat*(1-p_hat)/len(p_values)))

        pass_min = p_hat - pass_window
        pass_max = p_hat + pass_window
        return pass_min, pass_max


testlist = [
        'monobit_test',
        'frequency_within_block_test',
        'runs_test',
        'longest_run_ones_in_a_block_test',
        'binary_matrix_rank_test',
        'dft_test',
        'non_overlapping_template_matching_test',
        'overlapping_template_matching_test',
        'maurers_universal_test',
        'linear_complexity_test',
        'serial_test',
        'approximate_entropy_test',
        'cumulative_sums_test',
        'random_excursion_test',
        'random_excursion_variant_test',
        ]    

filepath = '/mnt/scapa4/Edward Lee/03-Projects/01-Stochastic Crystallisation/Code/RNGbot/analysis/string_generation/test.txt'
# filepath = r'Z:\group\Edward Lee\03-Projects\01-Stochastic Crystallisation\Code\RNGbot\analysis\randomness_testing\numbers\data.e'
# filepath = '/mnt/scapa4/Edward Lee/03-Projects/01-Stochastic Crystallisation/Code/RNGbot/analysis/randomness_testing/numbers/data.sqrt3'

bits = read_binary_expansion_file(filepath)
# print(bits)
np_bits = list(np.random.randint(0,2,100000))


chunks = get_chunks(np_bits, number_of_chunks = 55, chunk_length=None)
data = run_chunks(chunks)
results = collate_results(data)
stats = calc_stats(results)


for d in data:
    for t in d:
        print(t)

