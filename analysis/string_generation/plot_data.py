import numpy as np
import matplotlib.pyplot as plt

def plot_histograms(stats, colour='mediumseagreen'):
    tests = [i for i in stats if len(stats[i]['ps']) > 0]
    fig, axs = plt.subplots(len(tests), sharex='col', figsize=(6.3,9))

    
    # print(number_of_tests)
    # t2 = 'Substring length: {} sample size: {}\n\n'.format(stats['chunk size'], stats['chunks'])
    # t3 = 'Histograms of p-values generated by statistical tests of the NIST\n'
    # t4 = 'package for a binary string generated by crystallization of '
    # title = t2+t3+t4
    # fig.suptitle(title)

    for idx, test in enumerate(tests):

        axs[idx].bar(stats[test]['hist axis'], stats[test]['hist'], width=0.1, edgecolor='k', color=colour )
        ymax = int(max(stats[test]['hist']) *1.5)
        axs[idx].set_ylim(ymin=0, ymax = ymax)
        text_height = (ymax*0.8+max(stats[test]['hist']))/2
        if test == 'NonOverlappingTemplateMatching':
            fontsize = 6
        else:
            fontsize = 10
        axs[idx].text(0,text_height, test, fontsize=fontsize)
        axs[idx].text(0.45,text_height,'{}'.format(r'$p_{uniform}$'+'    ={0:.4f}'.format(stats[test]['uniformity'])))
        axs[idx].text(0.75, text_height,'pass rate= {0:.4f}'.format(stats[test]['pass rate']))
        axs[idx].minorticks_on()

        if idx == int(len(stats)/2) -1:
            axs[idx].set_ylabel('count', rotation=90)
    axs[idx].set_xlabel('p-value range')
    plt.subplots_adjust(wspace=0, hspace=0)
    return axs

    
def plot_pass_rates(stats, colour='mediumseagreen'):
    fig, ax = plt.subplots()
    xs = []
    ys = []
    tests = [i for i in stats if len(stats[i]['ps']) > 0]
    for test in tests:
        xs.append(test)
        ys.append(stats[test]['pass rate'])
    
    # t1 = 'Pass rates for statistical tests of the NIST package\n'
    # t2 = 'for a binary string generated by crystallization of '
    # fig.suptitle(t1+t2)
    plt.bar(xs, ys,width=0.8,edgecolor='k', color=colour)

    plt.hlines([1], [-1],[len(tests)],  linestyle='-.')
    plt.hlines([stats[test]['pass window'][0]], [-1],[len(tests)],  linestyle='-')
    plt.hlines([0.99], [-1],[len(tests)],  linestyle='--')
    plt.hlines([stats[test]['pass window'][0]], [-1],[len(tests)],  linestyle='-')
    plt.tick_params(axis='x', labelrotation=30, labelleft=True)
    # ax.axis['bottom'].label.set_text('right')
    plt.xticks(rotation=30, ha='right')
    plt.xlim(-1,len(tests))
    y_max = (stats[test]['pass window'][1] + 0.99) / 2
    y_min = (stats[test]['pass window'][0]+0.99)-1
    plt.ylim(y_min, y_max)
    plt.ylabel('Pass rate')
    return ax

def plot_uniformities(stats, colour='mediumseagreen'):
    fig, ax = plt.subplots()
    xs = []
    ys = []
    tests = [i for i in stats if len(stats[i]['ps']) > 0]
    for test in tests:
        xs.append(test)
        ys.append(stats[test]['uniformity'])

    # t1 = 'Uniformity of p-values for histograms of tests of the NIST\n'
    # t2 = 'package for a binary string generated by crystallization of '

    # ax.suptitle(t1+t2)
    plt.bar(xs, ys, width=0.8, edgecolor='k', color=colour)
    plt.xticks(rotation=30, ha='right')
    plt.xlim(-1,len(tests))
    plt.ylabel('Uniformity p-value')
    return ax
