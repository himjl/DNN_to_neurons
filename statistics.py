import numpy as np 

# Get metrics 
def sparsity(vec): 
    # Where vec is the a vector of image responses for a single neuron 

    vec = vec + 1e-32
    nsamps = float(len(vec))
    
    nsampsr = 1. / nsamps
    num = np.power(nsampsr * np.sum(vec), 2)
    denom = nsampsr * np.sum(np.power(vec, 2))
    a = np.true_divide(num, denom)
    s = np.true_divide((1. - a) , (1. - nsampsr))
    
    return s


# Fraction of images evoking responses above 50% of neuron's peak 
def frac_images(vec): 
    # Where vec is the a vector of image responses for a single neuron 
    m = np.max(vec)
    return np.mean(vec >= (0.5 * m))