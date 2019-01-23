import numpy as np 
import scipy.stats as ss 
import collections
import pandas as pd 
from tqdm import tqdm 

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

class dpdf: 
    def __init__(self, xsamples, binwidth, lb, ub): 
        xsamples = filter(lambda x: x >= lb and x<= ub, xsamples)
        px, left, right, center = discretize(xsamples, binwidth, lb, ub)
        
        self.lb = lb
        self.ub = ub 
        self.binwidth = binwidth
        
        self.x = center
        self.px = px 
        
        self.dist = {}
        for xi, pxi in zip(self.x, self.px): 
            self.dist[xi] = pxi
            
        
    def pdf(self, x): 
        try: 
            return self.dist[x]
        except KeyError: 
            return 0.
        
    def rvs(self, size = 1): 
        if size == 1: 
            return np.random.choice(self.x, p = self.px, size = None)
        else: 
            return np.random.choice(self.x, p = self.px, size = size)
    def plot(self, **kwargs): 
        plt.plot(self.x, self.px, **kwargs)
        
        
def get_bin_idx(samples, binwidth, lb = None, ub = None): 
    if np.any(samples > ub): 
        raise ValueError('Values lie outside of specified range [%0.1f, %0.1f] (max value found =%0.1f)'%(lb, ub, max(samples)))
        #ub = max(samples)
    if lb is None:
        lb = min(samples)
    if ub is None: 
        ub = max(samples)
    bins = np.arange(lb, ub+binwidth, binwidth)
    centers = bins[:-1] + 0.5 * (bins[1:] - bins[:-1])
    idx = np.digitize(samples, bins, right = True)
    idx[idx == 0] = 1 # Assign values equal to first left edge to the first bin (not the zeroth bin)
    idx = idx - 1
    
    return idx, centers
def digitize(samples, binwidth, lb = None, ub = None):
    # transforms samples to the value of the closest bin 
    idx, centers = get_bin_idx(samples, binwidth, lb = lb, ub = ub)
    return centers[idx]
    
    
    
class independent_joint_dpdf:
    def __init__(self, xsamples, binwidths, lbs, ubs): 
        self.dists = []
        
        self.xvals = []
        for d in range(xsamples.shape[1]):
            self.dists.append(dpdf(xsamples[:, d], binwidths[d], lbs[d], ubs[d]))
            self.xvals.append(self.dists[-1].x)
        
        
        # Calculate joint pmf of the discretized data 
        self.jpdf = np.ones(map(lambda n: len(n), self.xvals))
        for i, d in enumerate(self.dists): 
            p = d.px
            idx = [None for _ in self.dists]
            idx[i] = np.arange(len(p))
            idx = tuple(idx)
            self.jpdf = self.jpdf * p[idx]
            
        return 

    def pdf(self, x): 
        p = 1
        for xi, dist in zip(x, self.dists): 
            p*= dist.pdf(xi)
        return p        
        
    def rvs(self, size = 1, return_sample_index = False): 

        x = []
        for d in self.dists: 
            x.append(d.rvs(size = 1))
        if size == 1: 
            return np.array(x)
        else: 
            return np.array(x).transpose()
    
class joint_dpdf:
    def __init__(self, xsamples, binwidths, lbs, ubs): 

        d = xsamples.shape[1]
        assert len(binwidths) == len(lbs) == len(ubs) == d

        ineligible_idx = (np.zeros(len(xsamples))).astype(bool)
        for di in range(xsamples.shape[1]): 
            # Remove points lying outside of the user-specified support
            eligible_mask = map(lambda x: lbs[di] <= x <= ubs[di], xsamples[:, di])
            ineligible_mask = np.logical_not(eligible_mask)
            ineligible_idx+=ineligible_mask

        if np.sum(ineligible_idx) > 0: 
            print 'Found %d points outside of specified support (%0.2f of total)'%(np.sum(ineligible_idx), np.mean(ineligible_idx))
        if np.sum(ineligible_idx) >= xsamples.shape[0]: 
            raise ValueError('All sample points are outside of specified support')

        eligible_xsamples = xsamples[np.logical_not(ineligible_idx)]
        internal_idx_to_original_idx = np.where(np.logical_not(ineligible_idx))[0]
        self.xidx = []
        xvals = []
        for di in range(eligible_xsamples.shape[1]):
            idx, i_xvals = get_bin_idx(eligible_xsamples[:, di], binwidths[di], lb = lbs[di], ub = ubs[di])

            self.xidx.append(idx)
            xvals.append(i_xvals)

        self.xvals = xvals 
        self.xidx = np.array(self.xidx).transpose()

        # Estimate joint probability distribution 
        j = np.zeros(map(lambda x: len(x), xvals))
        self.jpdf = j
        self._raveled_jpdf_index_to_samples = collections.defaultdict(list)
        for i in tqdm(range(self.xidx.shape[0]), desc = 'estimating joint pmf'): 
            idx = tuple(self.xidx[i, :])
            self.jpdf[idx]+=1

            raveled_jpdf_idx = np.ravel_multi_index(idx, self.jpdf.shape)
            self._raveled_jpdf_index_to_samples[raveled_jpdf_idx].append(internal_idx_to_original_idx[i])

        self.jpdf = self.jpdf / float(self.xidx.shape[0])
        self._raveled_jpdf = np.ravel(self.jpdf)


    def get_bin_membership(self, x): 
        # assumes x is a single sample 
        idx = []
        for xi, bins in zip(x, self.xvals): 
            i_idx = np.digitize(xi, bins, right = True)
            i_idx[i_idx == 0] = 1 # Assign values equal to first left edge to the first bin (not the zeroth bin)
            i_idx = i_idx - 1
            idx.append(i_idx)
        return tuple(idx)

    def pdf(self, x): 

        for xi, bins in zip(x, self.xvals): 
            if xi < min(bins) or xi > max(bins): 
                return 0.
        else:
            sample_idx = self.get_bin_membership(x)
            return self.jpdf[sample_idx]
        
            sample_idx = []
            for xi, bins in zip(x, self.xvals): 
                sample_idx.append(np.argmin(np.abs(bins - xi))) # this is wrong. should follow histogram binning rule. 
            sample_idx = tuple(sample_idx)
            return self.jpdf[sample_idx]

    def rvs(self, size = 1, return_sample_index = False): 
        
        if size == 1:
            raveled_idx = np.random.choice(len(self._raveled_jpdf), p = self._raveled_jpdf, size = None)
            idx = np.unravel_index(raveled_idx, self.jpdf.shape)

            x = []
            for i, xdim in zip(idx, self.xvals): 
                x.append(xdim[i])

            if return_sample_index: 
                x = (x, np.random.choice(self._raveled_jpdf_index_to_samples[raveled_idx]))
        else: 
            raveled_idx = np.random.choice(len(self._raveled_jpdf), p = self._raveled_jpdf, size = size)
            idx = np.unravel_index(raveled_idx, self.jpdf.shape)
            
            x = []
            for i_idx, xdim in zip(idx, self.xvals): 
                x.append(xdim[i_idx])
            x = np.array(x).transpose()

            if return_sample_index: 
                x = x, [np.random.choice(self._raveled_jpdf_index_to_samples[i_idx]) for i_idx in raveled_idx]

        return x