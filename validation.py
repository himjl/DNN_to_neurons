
import collections
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 

import rust_results
reload(rust_results)
from tqdm import tqdm 
import scipy.stats as ss
import statistics
reload(statistics)
from statistics import sparsity, frac_images 
import pandas as pd 

    

def validate(df, Ftest): 

    def vec_stats(v): 
        d = {}
        d['mean'] = (np.mean(v))
        d['max'] = (np.max(v))
        d['min'] = (np.min(v))
        d['sparsity'] = (sparsity(v))
        d['frac'] = (frac_images(v))
        return d

    def get_test(Ftest, df): 
        df_test = collections.defaultdict(list)
        num_not_vis_responsive = 0
        for m in  tqdm(df.itertuples(), total = len(df), desc = 'statistics'): 
            alpha = m.alpha
            b = m.b
            v = alpha* Ftest[:, m.modelNeuronId] + b 


            if len(set(v)) == 1: 
                # Not visually responsive
                num_not_vis_responsive+=1
                continue
            s = vec_stats(v)
            for k in s: 
                df_test[k].append(s[k])
        print 'Num not visually responsive = %d; proportion = %0.2f'%(num_not_vis_responsive, num_not_vis_responsive / float(len(df)))
        return pd.DataFrame(df_test)

    def get_generated_population_response(Ftest, df):
        V = []
        for m in tqdm(df.itertuples(), desc = 'population response'): 
            v = m.alpha * Ftest[:, m.modelNeuronId] + m.b
            V.append(v)
        V = np.array(V).transpose()
        return V

    df_heldout = get_test(Ftest, df)
    N = get_generated_population_response(Ftest, df)

    readout(df_heldout)
    compare_rdms(Ftest, N)

    return 


def ks_test(df): 
    d = rust_results.get_IT_dists()


    pvals = {}
    x = np.array(df['sparsity'])
    pvals['sparsity'] = ss.kstest(x, d['sparsity'].cdf).pvalue

    x = np.array(df['max'])
    pvals['max'] = ss.kstest(x, d['max'].cdf).pvalue

    x = np.array(df['mean'])
    pvals['mean'] = ss.kstest(x, d['mean'].cdf).pvalue

    x = np.array(df['frac'])
    pvals['frac'] = ss.kstest(x, d['frac'].cdf).pvalue

    return pvals


def readout(df, label = None): 
    def discretize(samples, binwidth, lb = None, ub = None):
        if lb is None:
            lb = min(samples)
        if ub is None: 
            ub = max(samples)
        bins = np.arange(lb, ub+binwidth, binwidth)
        h, edges = np.histogram(samples, bins, density = False)
        h = h / float(len(samples))
        
        left = edges[0:-1]
        right = edges[1:]
        center = edges[0:-1] + (edges[1:] - edges[0:-1])*0.5
        return h, left, right, center


    def hist(vec, binwidth = None, lb = None, ub = None, label = None, metric = 'median', filled = True, **kwargs): 
        # Plots empirical pmf in bins 

        if metric is 'median':
            mstring ='median = %0.3f'%np.median(vec)
        elif metric is 'mean':
            mstring ='mean = %0.3f'%np.mean(vec)
        if label is None or label =='': 
            label = mstring
        else: 
            label = label + ' (%s)'%mstring
        h, left, right, center = discretize(vec, binwidth, lb = lb, ub = ub)
        if filled: 
            plt.bar(center, h, width = binwidth*0.6, label = label, **kwargs)
        else: 
            plt.plot(center, h, label = label, **kwargs)
        #plt.hist(vec, bins = bins, label = label, normed=True)

    def plot_pval(pval): 
        if pval < 0.001: 
            s = 'p = %0.1e'%pval
        else: 
            s = 'p = %0.3f'%pval
        ts.text(0.7, 0.7, s)


    pvals = ks_test(df)
    d = rust_results.get_IT_dists()

    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    plt.sca(ax[0, 1])
    plt.title('Max image evoked responses', weight = 'heavy')
 

    hist(df['max'], label = label, binwidth = 5, lb = 0, ub = 170, metric = 'median', alpha = 0.6)
    hist(d['max'].rvs(10000), binwidth = 5, lb = 0, ub = 170, metric = 'median', filled = False, color = sns.color_palette()[2], label = 'Rust and DiCarlo', lw = 1, marker = 'P')
    #xseq = np.linspace(0, 170, 10000)
    #plt.plot(xseq, d['max'].pdf(xseq), label = 'Rust and DiCarlo (med = %0.3f)'%(d['max'].median()))
    plot_pval(pvals['max'])
    plt.legend()
    plt.xlim([None, 170])
    plt.xlabel('Peak firing rate (Hz)')
    
    plt.sca(ax[0, 0])
    _ = plt.title('Grand mean evoked responses', weight = 'heavy')
    _ = hist(df['mean'], label = label, binwidth = 2.5, lb = 0, ub = 70, metric = 'median', alpha = 0.6)
    hist(d['mean'].rvs(10000), binwidth = 2.5, lb = 0, ub = 70, metric = 'median', filled = False, color = sns.color_palette()[2], label = 'Rust and DiCarlo', lw = 1, marker = 'P')
    #xseq = np.linspace(0, 70, 10000)
    #_ = plt.plot(xseq, d['mean'].pdf(xseq), label = 'Rust and DiCarlo (med = %0.3f)'%(d['mean'].median()))
    plot_pval(pvals['mean'])
    _ = plt.legend()
    _ = plt.xlim([None, 70])
    _ = plt.xlabel('Grand mean firing rate (Hz)')
    
    plt.sca(ax[1, 0])
    _ = plt.title('Sparsity', weight = 'heavy')
    _ = hist(df['sparsity'], label = label, metric = 'mean', binwidth = 0.1, lb = 0, ub = 1, alpha = 0.6)
    hist(d['sparsity'].rvs(10000), binwidth = 0.1, lb = 0, ub = 1, metric = 'mean', filled = False, color = sns.color_palette()[2], label = 'Rust and DiCarlo', lw = 1, marker = 'P')
    #xseq = np.linspace(0, 1, 10000)
    #_ = plt.plot(xseq, d['sparsity'].pdf(xseq), label = 'Rust and DiCarlo (mean = %0.3f)'%(d['sparsity'].mean()))
    plot_pval(pvals['sparsity'])
    _ = plt.legend()
    _ = plt.xlabel('Sparsity')
    
    plt.sca(ax[1, 1])
    _ = plt.title('Fraction', weight = 'heavy')
    _ = hist(df['frac'], label =label, metric = 'mean',  binwidth = 0.05, lb = 0, ub = 1, alpha = 0.6)
    hist(d['frac'].rvs(10000), binwidth = 0.05, lb = 0, ub = 1, metric = 'mean', filled = False, color = sns.color_palette()[2], label = 'Rust and DiCarlo', lw = 1, marker = 'P')
    #xseq = np.linspace(0, 1, 10000)
    #_ = plt.plot(xseq, d['frac'].pdf(xseq), label = 'Rust and DiCarlo (mean = %0.3f)'%(d['frac'].mean()))
    plot_pval(pvals['frac'])
    _ = plt.legend()
    _ = plt.xlabel('Fraction of images evoking firing rates\nthat exceed 50% of the peak')

def compare_rdms(pop_native, pop_synth): 

    rdm_native = ts.rdm(pop_native)
    rdm_synth = ts.rdm(pop_synth)

    plt.figure(figsize = (4, 4))
    ts.unityline()

    v1, v2 = rdm_native.ravel(), rdm_synth.ravel()
    plt.plot(v1, v2, '.', alpha = 0.5, ms = 1)
    ev = ts.explained_variance(v1, v2)
    plt.xlabel('Native model population')
    plt.ylabel('Shifted and rescaled population')
    plt.title('Native vs. Generated RDM', weight= 'heavy')
    ts.text(0.2, 0.8, 'ev = %0.2f'%ev)