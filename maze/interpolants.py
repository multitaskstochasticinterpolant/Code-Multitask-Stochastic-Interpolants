import torch
import torch.nn as nn
from torch.func import vmap


class Interpolant:
    def _single_xt(self, x0, x1, alpha):
        return alpha*x0 + (1 - alpha)*x1
        
    def alpha(self, alpha):
        return alpha
        
    def dotalpha(self, alpha):
        return -1.0 + 0*alpha
        
    def beta(self, alpha):
        return 1 - alpha
        
    def dotbeta(self, alpha):
        return 1.0 + 0*alpha
        
    def _single_dtxt(self, x0, x1):
        return x0 - x1
    
    def xt(self, x0, x1, alpha):
        #Repeat along the two coordinates
        if len(alpha.shape) == 2:
            alpha = alpha[...,np.newaxis].repeat(1, 1, 2)
        # elif len(alpha.shape) == 3:
        #     alpha = alpha[...,np.newaxis].repeat(1, 1, 2)
        return vmap(self._single_xt, in_dims=(0, 0, 0))(x0,x1,alpha) #Vectorize over the batch size
    
    def dtxt(self, x0, x1):
        return vmap(self._single_dtxt, in_dims=(0, 0))(x0,x1)
