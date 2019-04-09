import numpy as np
'''
https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
'''
def Batchnorm(x,alpha,beta):
    mean =np.mean(x,axis=0)
    var= np.var(x,axis=0)
    x_norm = (x-mean)/np.sqrt(var+1e-8)
    out = alpha*x_norm +beta
    cache = (x,x_norm,mean,var,alpha,beta)
    return out,cache,mean,var

def batchnorm_backward(dout, cache):
    x, x_norm, mean, var, alpha, beta = cache
    dalpha = np.sum(dout*x_norm,axis=0)
    dbeta= np.sum(dout,axis=0)

    return dalpha,dbeta






