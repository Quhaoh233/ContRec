# credit: https://github.com/raywang4/DispLoss
import torch as th

def disp_loss(self, z): # Dispersive Loss implementation (InfoNCE-L2 variant)
    z = z.reshape((z.shape[0],-1)) # flatten
    diff = th.nn.functional.pdist(z).pow(2)/z.shape[1] # pairwise distance
    diff = th.concat((diff, diff, th.zeros(z.shape[0]).cuda()))  # match JAX implementation of full BxB matrix
    return th.log(th.exp(-diff).mean()) # calculate loss