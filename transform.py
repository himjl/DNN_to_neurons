import tasso as ts 
import numpy as np 
import pandas as pd 
from tqdm import tqdm


import statistics
reload(statistics)
from statistics import sparsity, frac_images

import rust_results
reload(rust_results)

import sampling_methods 
reload(sampling_methods)

def obtain_mapping(Ftrain = None, df_projections = None, method = 'sequential_histogram_matching', num_simulated_neurons = 10000): 
    
    if Ftrain is None and df_projections is None: 
        raise ValueError('Must give either Ftrain or df_projections')
    if Ftrain is not None and df_projections is not None:
        raise ValueError('Must give ONE of Ftrain or df_projections')



    if method == 'sequential_histogram_matching':
        # Generate a bunch of random projections, with matched mean distributions to IT. 
        # Then, filter the projections for the max, sparsity, and frac distributions (in order) so that they match the true histograms (at some resolution)
        # This relies on the independence assumption between the four statistics (?)
        if df_projections is None: 
            df_projections = sample_IT_projections(Ftrain)
        df_mapping = sampling_methods.sequential_histogram_matching(df_projections, total_samples = num_simulated_neurons)
    elif method == 'metropolis_hastings':
        df_mapping = sampling_methods.metropolis_hastings(Ftrain)
    elif method == 'empirical_rejection_sampling':
        # Assume the four target distributions are independent. 
        # Generate random projections. 
        # Use this as a generating distribution for rejection sampling (with the probability of any sample being 1 / N) against the estimated rust targets
        if df_projections is None: 
            df_projections = sample_IT_projections(Ftrain)
        df_mapping = sampling_methods.empirical_rejection_sampling(df_projections, max_iter = 500000, nsamps = num_simulated_neurons)
    elif method == 'discrete_rejection_sampling':
        if df_projections is None: 
            df_projections = sample_IT_projections(Ftrain)
        df_mapping = sampling_methods.discrete_rejection_sampling(df_projections, max_iter = 500000, nsamps = num_simulated_neurons)

    return df_mapping

