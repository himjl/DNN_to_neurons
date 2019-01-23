import numpy as np 
import pandas as pd
import tasso as ts 


import rust_results 
reload(rust_results)
from rust_results import get_IT_dists

from tqdm import tqdm 
import statistics 
reload(statistics)
from statistics import sparsity, frac_images
import collections

max_possible_meanHz = 60
max_possible_maxHz = 125 
def get_max_prob_beta(beta_dist):
    a, b = beta_dist.args
    loc = beta_dist.kwds['loc']
    scale = beta_dist.kwds['scale']
    
    mode_internal = (a - 1)/float(a + b - 2)
    mode = mode_internal*scale + loc
    return beta_dist.pdf(mode)
# Design equation for linearly transforming vector to have mean mu, max firing maxHz
def get(vec, mu, maxHz): 
    
    n2 = maxHz
    v1 = min(vec)
    v2 = max(vec)
    vbar = np.mean(vec)
    
    alpha = (n2 - mu)/float(v2 - vbar)
    b = mu - alpha * vbar 
    
    return alpha, b

def mu_range(vec, maxHz): 
    alpha = maxHz / float(max(vec) - min(vec))
    mu_min = alpha * np.mean(vec) 
    return mu_min, maxHz

def max_range(vec, mu): 
    v = vec
    v2 = max(vec)
    v1 = min(vec)
    vbar = np.mean(vec)
    n2_max = mu * (max(v) - min(v)) / (np.mean(v) - min(v))
    n2_min = 0.02 + mu * (max(v) - min(v)) / (0.5 * np.mean(v) - 0.5 * min(v) + 0.5 * (max(v) - min(v)))

    ub_max = n2_max # So that the lowest activation must be above zero. 
    lb_max = max(mu, n2_min) 
    # Lower bound so that the max is 1) above the mean, and 2) above the value that would have the min > 0.5 max; 
    # i.e. that would guarantee the image fraction is 1.
    return lb_max, ub_max


def sample_IT_projections(Ftrain, n_projections = 100, n_projections2 = 25): 

    dist_mean = rust_results.get_Rust_IT_mean_distribution()
    #n_grid = 50
    

    df = ts.d()

    for i in tqdm(range(Ftrain.shape[1]), desc = 'Projecting native neurons'):
        
        vec = Ftrain[:, i]
        if len(set(vec)) == 1: 
            continue
        base_sparsity = sparsity(vec)
        base_fraction = frac_images(vec)
        
        for j in range(n_projections): 
            mu = dist_mean.rvs()
            lb, ub = max_range(vec, mu) # Gives bounds that satisfy that the neuron will be nonzero; and that the min value will be less than 0.5 the max
            ub = min(ub, max_possible_maxHz)

            for maxHz in np.linspace(lb, ub, n_projections2): 
            
                alpha, b = get(vec, mu, maxHz)
                v = alpha * vec + b
                v = v + 1e-32

                df['modelNeuronId'].append(i)
                df['alpha'].append(alpha)
                df['b'].append(b)
                df['mean'].append(np.mean(v))
                df['max'].append(np.max(v))
                df['min'].append(np.min(v))
                df['lb_max'].append(lb)
                df['ub_max'].append(ub)
                df['design_mean'].append(mu)
                df['design_max'].append(maxHz)
                df['base_sparsity'].append(base_sparsity)
                df['base_fraction'].append(base_fraction)
                df['sparsity'].append(sparsity(v))
                df['frac'].append(frac_images(v))

    df = pd.DataFrame(df)
    df = df.dropna()

    return df 


def sequential_histogram_matching(df, total_samples = 10000): 
    # Assumes df already has target mean distribution

    def bin_prob(dist, lb, ub): 
        return dist.cdf(ub) - dist.cdf(lb)

    def filter_via_binning(df, target_dist, v = 'max', binwidth = 1, total_samples = 10000):
        # Sample with replacement at each bin to match overall ratio    
        
        nbins = np.ceil(float(max(df[v]) - min(df[v])) / float(binwidth))
        print 'nbins = %d'%nbins
        bin_membership = pd.cut(df[v], nbins, labels = None, retbins = False, include_lowest = True)
        ef = df.copy()
        ef['bin'] = bin_membership
        ef = ef.reset_index(drop = True)
        
        sampIdxs = []

        for b, bf in tqdm(ef.groupby('bin'), desc = 'matching %s bins'%v):
            lb = b.left
            ub = b.right
            target_proportion = bin_prob(target_dist, lb, ub)
            n = int(np.round(target_proportion* total_samples) )
            if n == 0 or len(bf) == 0: 
                continue
            sampIdxs.extend(np.random.choice(bf.index.values, size = n, replace = True))
        print len(sampIdxs)
        df_binfiltered = ef.loc[sampIdxs]
        return df_binfiltered

    dists = get_IT_dists()
    #df_binfiltered = filter_via_binning(df, dist, v = 'mean', binwidth = 0.5, total_samples = 10000)
    df_binfiltered = df 
    
    df_binfiltered = filter_via_binning(df_binfiltered, dists['frac'], v = 'frac', binwidth = 0.0005, total_samples = total_samples)
    df_binfiltered = filter_via_binning(df_binfiltered, dists['sparsity'], v = 'sparsity', binwidth = 0.005, total_samples = total_samples)
    df_binfiltered = filter_via_binning(df_binfiltered, dists['max'], v = 'max', binwidth = 0.5, total_samples = total_samples)
    return df_binfiltered

def metropolis_hastings(Ftrain, nsamps = 1000, n_burn_in = 10000): 

    d = get_IT_dists()
    def prob_params(alpha, b, vec): 
        n = alpha * vec + b
        
        mx, mu, frac, s = stats(n)
        
        p_mx = d['max'].pdf(mx)
        p_mu = d['mean'].pdf(mu)
        p_frac = d['frac'].pdf(frac)
        p_s = d['sparsity'].pdf(s)
        
        pr = p_mx * p_mu * p_frac * p_s # assume independence
        
        return pr

    def stats(n):
        
        mx = max(n)
        mu = np.mean(n)
        frac = frac_images(n)
        s = sparsity(n)
        
        return mx, mu, frac, s

    def sample_params(): 
        modelNeuronId = np.random.choice(Ftrain.shape[1])
        vec = Ftrain[:, modelNeuronId]
        mu = np.random.rand() * max_possible_meanHz # sample_mean()
        lb, ub = max_range(vec, mu)
        ub = min(ub, max_possible_maxHz)
        maxHz = np.random.rand() * (ub - lb) + lb
        alpha, b = get(vec, mu, maxHz)
        return alpha, b, vec, modelNeuronId

    alpha, b, vec, modelNeuronId = sample_params()
    plast = prob_params(alpha, b, vec)


    df = ts.d()

    for i in tqdm(range(nsamps + n_burn_in)):
        
        # Get next proposal
        alpha_proposal, b_proposal, vec_proposal, modelNeuronId_proposal = sample_params()
        
        pcur = prob_params(alpha_proposal, b_proposal, vec_proposal)
        if plast == 0: 
            raise Exception
        if np.log10(np.random.rand()) < (np.log10(pcur) - np.log10(plast)): 
            # Accept
            alpha = alpha_proposal
            b = b_proposal
            vec = vec_proposal 
            modelNeuronId = modelNeuronId_proposal
            plast = pcur 
            
        if i < n_burn_in: 
            continue
        # Get stats
        mx, mu, frac, s = stats(alpha * vec + b)
        df['max'].append(mx)
        df['mean'].append(mu)
        df['frac'].append(frac)
        df['sparsity'].append(s)
        df['alpha'].append(alpha)
        df['b'].append(b)
        df['modelNeuronId'].append(modelNeuronId)

    df = pd.DataFrame(df)


    return df

def empirical_rejection_sampling(df, max_iter = 5000000, nsamps = 1000): 

    

    d = get_IT_dists()
    max_prob_target = get_max_prob_beta(d['max']) * get_max_prob_beta(d['mean']) * get_max_prob_beta(d['frac']) * get_max_prob_beta(d['sparsity'])


    prob_generator = 1./float(len(df))

    # Get M for rejection sampling >= max(prob_target(x) / prob_generator(x))
    M = max_prob_target / prob_generator
    
    print 'Rejection sampling M: %0.3f'%M

    filter_idxs = []
    with tqdm(total = nsamps, desc = 'obtained samples') as pbar:
        for _ in range(max_iter):
            i = np.random.choice(len(df))

            mx = df.iloc[i]['max'] 
            mu = df.iloc[i]['mean'] 
            f = df.iloc[i]['frac'] 
            s = df.iloc[i]['sparsity']

            prob_target = d['max'].pdf(mx) * d['mean'].pdf(mu) * d['frac'].pdf(f) * d['sparsity'].pdf(s)
            keep_prob = prob_target / float(M * prob_generator)

            if np.random.rand() < keep_prob: 
                filter_idxs.append(i)
                pbar.update(1)
            if len(filter_idxs) >= nsamps: 
                break

    df_filtered = df.iloc[filter_idxs].copy()
    
    return df_filtered


def discrete_rejection_sampling(df, max_iter = 500000, nsamps = 1000): 
    import discretize_utils
    reload(discretize_utils)
    from discretize_utils import independent_joint_dpdf, joint_dpdf

    # Transform to (assumedly independent) discrete distributions 
    d = get_IT_dists()
    target_mean_samps = d['mean'].rvs(10000)
    target_max_samps = d['max'].rvs(10000)
    target_sparsity_samps = d['sparsity'].rvs(10000)
    target_frac_samps = d['frac'].rvs(10000)

    mean_binwidth = 2.5 
    max_binwidth = 5
    sparsity_binwidth = 0.1 
    frac_binwidth = 0.05 

    binwidths = [mean_binwidth, max_binwidth, sparsity_binwidth, frac_binwidth] 
    lbs = [0, 0, 0, 0]
    ubs = [max_possible_meanHz, max_possible_maxHz, 1, 1]
    if False:
        binwidths = [mean_binwidth, sparsity_binwidth] 
        lbs = [np.min(target_mean_samps), np.min(target_sparsity_samps),]
        ubs = [np.max(target_mean_samps), np.max(target_sparsity_samps)]
        print lbs, ubs

        if True:
            binwidths = [mean_binwidth, max_binwidth, sparsity_binwidth, frac_binwidth] 
            lbs = [np.min(target_mean_samps), np.min(target_max_samps), np.min(target_sparsity_samps), np.min(target_frac_samps)]
            ubs = [np.max(target_mean_samps), np.max(target_max_samps), np.max(target_sparsity_samps), np.max(target_frac_samps)]
            print lbs, ubs

    # Generate discrete target distribution 
    xsamp_target = np.array((
        target_mean_samps,
        target_max_samps,
        target_sparsity_samps,
        target_frac_samps,
        )).transpose()

    if True: 
        target_dist = independent_joint_dpdf(xsamp_target, binwidths, lbs, ubs)
    if False: 
        # Remove impossible samples 
        valid_mask = xsamp_target[:, 0] < xsamp_target[:, 1] 
        xsamp_target = xsamp_target[valid_mask, :]
        print 'Keeping:', np.sum(valid_mask)
        target_dist = joint_dpdf(xsamp_target, binwidths, lbs, ubs)

    # Transform source samples to joint discrete distribution
    if False: 
        xsamp = np.array((
            df['mean'], 
            df['max'], 
            df['sparsity'], 
            df['frac']
            )).transpose()
    if True: 
        xsamp = np.array((
        np.random.rand(10000)*(ubs[0] - lbs[0]) + lbs[0], 
        np.random.rand(10000)*(ubs[1] - lbs[1]) + lbs[1], 
        np.random.rand(10000)*(ubs[2] - lbs[2]) + lbs[2], 
        np.random.rand(10000)*(ubs[3] - lbs[3]) + lbs[3],
        )).transpose()
    if False: 
        source_dist = joint_dpdf(xsamp, binwidths, lbs, ubs)
    if True: 
        source_dist = target_dist #independent_joint_dpdf(xsamp, binwidths, lbs, ubs)
    # Get M for rejection sampling >= max(prob_target(x) / prob_generator(x))
    #return source_dist, target_dist

    # Todo: replace with relaxation of joint matching to marginal matching via transformation of generator joint 
    if True: 
        to = np.array(target_dist.jpdf) 
        so = np.array(source_dist.jpdf )
        s = so[so > 0]
        t = to[so > 0]
        if np.any(so<=0): 
            nignore = len(np.where(so <= 0)[0])
            print 'Throwing away %d entries (%0.3f probability mass) from target, due to zero support in these regions from generator'%(nignore, np.sum(to[so <= 0]))
    else: 
        t = target_dist.jpdf 
        s = source_dist.jpdf

    M = np.max(np.true_divide(t, s)) + 1e-32
    
    print 'Rejection sampling M: %0.3f'%M
    print 'Expected iterations to %d samples: %0.1f'%(nsamps, (M * nsamps))
    

    filter_idxs = []

    if True: 
        df_sim = ts.d()
    with tqdm(total = nsamps, desc = 'obtained samples') as pbar:
        for i_iter in range(max_iter):

            if True: 
                xsample = source_dist.rvs(return_sample_index = False)
                prob_target = target_dist.pdf(xsample)
                prob_generator = source_dist.pdf(xsample)
                keep_prob = prob_target / float(M * prob_generator)

                if np.random.rand() < keep_prob: 
                    df_sim['mean'].append(xsample[0])
                    df_sim['max'].append(xsample[1])
                    df_sim['sparsity'].append(xsample[2])
                    df_sim['frac'].append(xsample[3])
                    pbar.update(1)
                if len(df_sim['mean']) >= nsamps: 
                    print 'Iterations to completion: %d'%(i_iter)
                    break
            if False: 
                xsample, data_idx = source_dist.rvs(return_sample_index = True)
                prob_target = target_dist.pdf(xsample)
                prob_generator = source_dist.pdf(xsample)
                keep_prob = prob_target / float(M * prob_generator)

                if np.random.rand() < keep_prob: 
                    filter_idxs.append(data_idx)
                    pbar.update(1)
                if len(filter_idxs) >= nsamps: 
                    print 'Iterations to completion: %d'%(i_iter)
                    break

    if False: 
        df_filtered = df.iloc[filter_idxs].copy()
    if True: 
        df_filtered = pd.DataFrame(df_sim)
    
    return df_filtered, target_dist, source_dist