# Approximates the IT single unit distribution on four statistics given in Rust and DiCarlo

# 1. Expected max firing rate (expectation over neurons; max over images)
# 2. Expected grand mean firing rate (expectation over neurons, grand mean over images)
# 3. Expected sparsity (expectation over neurons)
# 4. Expected fraction (expectation over neurons)

import scipy.stats as ss 
import numpy as np 
import statistics
reload(statistics)
import matplotlib.pyplot as plt 

# Approximation of sparsity distribution of IT; b = 3.6, scale = 1; loc = 0, a = 1.632558
def get_Rust_IT_sparsity_distribution(): 
    b = 3.6
    design_mean = 0.312
    scale = 1
    loc = 0 
    a = np.true_divide((design_mean - loc)*b, scale - (design_mean - loc))

    x = np.linspace((0+loc), (1+loc)*scale, 10000)

    dist_sparsity = ss.beta(a, b, loc = loc, scale = scale)
    return dist_sparsity


# Approximation of max firing rate distribution of IT; b = 10, scale = 125; loc = 0, a = 2.0308
def get_Rust_IT_max_distribution(): 
    b = 10
    design_mean = 21.1
    scale = 125
    loc = 1 #design_mean - a * scale / float(a + b)#-scale * mean + (float(a) / float(a + b))
    a = np.true_divide((design_mean - loc)*b, scale - (design_mean - loc))

    x = np.linspace((0+loc), (1+loc)*scale, 10000)

    dist = ss.beta(a, b, loc = loc, scale = scale)
    return dist 

# Approximation of grand mean firing rate distribution of IT; b = 20, scale = 60; loc = 0, a = 2.2222
def get_Rust_IT_mean_distribution(): 

    b = 20
    design_mean = 6.
    scale = 60
    loc = 0 #design_mean - a * scale / float(a + b)#-scale * mean + (float(a) / float(a + b))
    a = np.true_divide((design_mean - loc)*b, scale - (design_mean - loc))

    x = np.linspace((0+loc), (1+loc)*scale, 10000)

    dist_mean = ss.beta(a, b, loc = loc, scale = scale)
    return dist_mean

# Approximation of fraction images driving neuron above 0.5*max(n) distribution of IT; 
def get_Rust_IT_fraction_distribution(): 
    b = 11

    design_mean = 0.094
    scale = 1
    loc = 0 #design_mean - a * scale / float(a + b)#-scale * mean + (float(a) / float(a + b))
    a = np.true_divide((design_mean - loc)*b, scale - (design_mean - loc))

    x = np.linspace((0+loc), (1+loc)*scale, 10000)

    dist_frac = ss.beta(a, b, loc = loc, scale = scale)
    return dist_frac 

def visualize_rust_results():
    d = get_IT_dists()

    fig, ax = plt.subplots(2, 2, figsize = (8, 8))

    # Mean
    plt.sca(ax[0, 0])
    x = np.linspace(0, 60, 10000)
    p = d['mean'].pdf(x)
    plt.axvline(d['mean'].median(), lw = 2, ls = ':', color = 'black', alpha = 0.5)
    plt.plot(x, p)
    plt.title('Approximation of IT\ngrand mean firing rate distribution', weight = 'heavy')
    plt.xlabel('Mean firing rate (Hz)')
    plt.legend()
    plt.xlim([0, 70])
    plt.text(20, 0.06, 'median = %0.1f'%d['mean'].median())

    # Max
    plt.sca(ax[0, 1])
    x = np.linspace(0, 160, 10000)
    p = d['max'].pdf(x)
    plt.plot(x, p)
    plt.axvline(d['max'].median(), lw = 2, ls = ':', color = 'black', alpha = 0.5)

    plt.title('Approximation of IT\nmax firing rate distribution', weight = 'heavy')
    plt.xlabel('Peak firing rate (Hz)')
    plt.text(50, 0.015, 'median = %0.1f'%d['max'].median())
    plt.xlim([0, 160])

    # Sparsity
    plt.sca(ax[1, 0])
    x = np.linspace(d['sparsity'].a, d['sparsity'].b, 1000)
    p = d['sparsity'].pdf(x)
    plt.plot(x, p)
    plt.axvline(d['sparsity'].mean(), lw = 2, ls = ':', color = 'black', alpha = 0.5)
    plt.title('Approximation of IT\nsparsity distribution', weight = 'heavy')
    plt.xlabel('sparsity')
    plt.text(0.6, 1.5, 'mean = %0.3f'%d['sparsity'].mean())


    # Fraction 

    def bin_prob(dist, lb, ub): 
        return dist.cdf(ub) - dist.cdf(lb)


    plt.sca(ax[1, 1])

    plt.axvline(d['frac'].median(), lw = 2, ls = ':', color = 'black', alpha = 0.5)

    binwidth = 0.05
    lefts = np.arange(0, 1, binwidth)
    rights = np.arange(binwidth, 1+binwidth, binwidth)
    p = []
    x = []

    for l, r in zip(lefts, rights): 
        p.append(bin_prob(d['frac'], l, r))
        x.append(l)
        
    plt.bar(x, p, width = binwidth/1.5, align = 'edge')

    plt.title('Approximation of IT\nfraction distribution', weight = 'heavy')
    plt.xlabel('Fraction images driving neuron above 0.5*max(neuron)')
    plt.legend()
    plt.text(0.2, 0.25, 'mean = %0.3f'%d['frac'].mean())

    plt.tight_layout()
    return 


def get_IT_dists(): 
    d = {}
    d['max'] = get_Rust_IT_max_distribution()
    d['mean'] = get_Rust_IT_mean_distribution()
    d['sparsity'] = get_Rust_IT_sparsity_distribution()
    d['frac'] = get_Rust_IT_fraction_distribution()

    return d 

