import torch
import numpy as np
from torch.distributions import MultivariateNormal

class BaseDistribution:
    """
    Class reprensenting data from base distribution.
    """
    def __init__(self, mean=torch.tensor([0.]), cov=torch.eye(1), device='cpu'):
        self.device = device
        self.mean = mean.to(device)
        self.covariance = cov.to(device)
        self.distribution = MultivariateNormal(self.mean, self.covariance)
    
    def sample(self, l, n=1, actions: bool = False):
        """
        Draws $n$ samples from the Gaussian distribution.   
        """
        if not actions:
            return self.distribution.sample((n, 1, l, 2)).squeeze(dim=-1).to(self.device)
        elif actions:
            return self.distribution.sample((n, 2, l, 2)).squeeze(dim=-1).to(self.device)
    
    def log_prob(self, x):
        """
        Evaluates the log probability of given samples $x$ under the distribution. 
        """
        return self.distribution.log_prob(x).to(device)

def target(n: int, array, l:int, inter: int=6, device='cpu'):
    """
    Generate data from target distributions.
    Args:
        n: (int). Batch size.
        l: (int). Length of the time serie.
        device: (str). Processor on which to do the computations
    """
    T, dim = array.shape
    if l > T:
        raise ValueError("Slice length l cannot be greater than the total time points T.")
    if n * l > T:
        raise ValueError("The total number of elements in all slices cannot exceed the total elements in the input array.")
    # Initialize the output array
    slices = np.zeros((n, 1, l//inter, 2))
    # Randomly sample starting indices for the slices
    # Ensure the starting index allows for a full slice of length l
    start_indices = np.random.choice(T - l + 1, size=n, replace=False)
    # Extract slices
    for i, start_idx in enumerate(start_indices):
        slices[i, 0] = array[start_idx:start_idx + l:inter, :2]
    return torch.cat((torch.tensor(slices[...,0]).float().to(device)[...,None], torch.tensor(slices[...,1]).float().to(device)[...,None]), axis = -1)